import asyncio
import os
from typing import Dict, Any, List
from flask import Flask, request as flask_request, jsonify, send_from_directory
from python_a2a.mcp import FastMCP, text_response, create_fastapi_app
from python_a2a import AgentCard, A2AServer, AgentSkill
import google.generativeai as genai
import aiohttp
import json
import re
import uuid
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This will print to console
        logging.FileHandler('coordinator.log')  # This will also save to a file
    ]
)

# Load environment variables from .env file
load_dotenv()

class CoordinatorAgent:
    def __init__(self):
        self.mcp_server = FastMCP("Coordinator MCP Server")
        self.ai_provider = self._setup_gemini()
        self.agent_card = self._create_agent_card()
        self.a2a_server = None
        self.server = None
        
        # Register MCP tools
        self._register_mcp_tools()
    
    def _setup_gemini(self):
        """Setup Gemini AI provider"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    
    def _create_agent_card(self) -> AgentCard:
        """Create agent card programmatically using python-a2a"""
        return AgentCard(
            name="coordinator",
            description="AI agent that can handle general tasks and coordinate with specialized agents when needed. Responds in Markdown.",
            url="http://localhost:3001",
            version="1.0.0",
            provider="Coordinator AI",
            capabilities={
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True
            },
            default_input_modes=["text/plain"],
            default_output_modes=["text/markdown", "text/html"],
            skills=[
                AgentSkill(
                    name="general_conversation",
                    description="Handle general conversations and questions. Responds in Markdown.",
                    tags=["conversation", "general", "assistance", "markdown"],
                    examples=[
                        "Answer general questions in Markdown format",
                        "Engage in casual conversation, formatted in Markdown"
                    ],
                    input_modes=["text/plain"],
                    output_modes=["text/markdown", "text/html"]
                ),
                AgentSkill(
                    name="task_coordination",
                    description="Coordinate with specialized agents for complex tasks. Provides updates in Markdown.",
                    tags=["coordination", "delegation", "workflow", "markdown"],
                    examples=[
                        "Delegate content creation to curator agent, report in Markdown",
                        "Coordinate multi-step tasks, provide Markdown updates"
                    ],
                    input_modes=["text/plain"],
                    output_modes=["text/markdown", "text/html"]
                )
            ]
        )
    
    def _register_mcp_tools(self):
        """Register MCP tools using FastMCP"""
        
        async def actual_handle_request(instance, request_str: str) -> Dict[str, Any]:
            try:
                direct_handling_prompt = f"""Analyze the following request: "{request_str}"
                Can you handle this request directly with a general conversational answer? 
                Your answer MUST be in Markdown format.
                Respond with a JSON object: {{"can_handle_directly": true/false, "direct_response_if_any_markdown": "your Markdown answer if true, else null"}}."""
                
                ai_direct_check_response = await asyncio.to_thread(instance.ai_provider.generate_content, direct_handling_prompt)
                direct_check_text = ai_direct_check_response.text
                can_handle_directly = False
                direct_response_markdown = None

                try:
                    json_match_direct = re.search(r"```json\s*([\s\S]*?)\s*```", direct_check_text, re.IGNORECASE)
                    if json_match_direct:
                        direct_decision_json_text = json_match_direct.group(1)
                    else:
                        direct_decision_json_text = direct_check_text.strip()
                    
                    direct_decision = json.loads(direct_decision_json_text)
                    can_handle_directly = direct_decision.get("can_handle_directly", False)
                    if can_handle_directly:
                        direct_response_markdown = direct_decision.get("direct_response_if_any_markdown")
                        if not direct_response_markdown:
                             direct_answer_prompt = f'Provide a direct answer to: "{request_str}". Your answer MUST be in Markdown format.' 
                             direct_answer = await asyncio.to_thread(instance.ai_provider.generate_content, direct_answer_prompt)
                             direct_response_markdown = direct_answer.text

                except json.JSONDecodeError:
                    logging.info(f"Could not parse direct handling JSON: {direct_check_text}. Assuming cannot handle directly.")
                    can_handle_directly = False
                except Exception as e:
                    logging.error(f"Error in direct handling check: {e}. Assuming cannot handle directly.")
                    can_handle_directly = False

                if can_handle_directly and direct_response_markdown:
                    logging.info(f"Handling request directly with Markdown: {request_str}")
                    return {
                        "success": True,
                        "handled_directly": True,
                        "response": direct_response_markdown
                    }

                logging.info(f"Request '{request_str}' not handled directly, attempting delegation.")
                available_agents_cards = await instance._get_available_agents()
                if not available_agents_cards:
                    return {"success": False, "error": "No specialized agents available for delegation."}

                agent_skill_info_for_prompt = []
                for agent_card_summary in available_agents_cards:
                    agent_info = {
                        "agent_name": agent_card_summary.get("name"),
                        "agent_description": agent_card_summary.get("description"),
                        "skills": []
                    }
                    full_agent_card = await instance._get_agent_card(agent_card_summary.get("name"))
                    if full_agent_card and full_agent_card.get("skills"):
                         for skill_detail in full_agent_card.get("skills"):
                            agent_info["skills"].append({
                                "skill_name": skill_detail.get("name"),
                                "skill_description": skill_detail.get("description"),
                                "skill_examples": skill_detail.get("examples"), 
                                "skill_input_modes": skill_detail.get("input_modes")
                            })
                    if agent_info["skills"]:
                        agent_skill_info_for_prompt.append(agent_info)

                if not agent_skill_info_for_prompt:
                    return {"success": False, "error": "No agents with usable skills found for delegation."}

                delegation_prompt = f"""You are a coordinator AI. Your task is to determine if a user request should be delegated to a specialized agent (curator or evaluator) and to prepare the necessary parameters for that agent.

User Request: "{request_str}"

Available Specialized Agents and their general capabilities:
{json.dumps(agent_skill_info_for_prompt, indent=2)}

Based on the User Request and agent capabilities:
1.  Decide if the request is best handled by the 'curator' or 'evaluator' agent.
2.  If delegation is appropriate, extract the necessary parameters from the User Request.

Output Format:
Respond ONLY with a single valid JSON object with the following keys:
1.  "chosen_agent_name": "curator", "evaluator", or null if no delegation is appropriate.
2.  "parameters_for_agent": A JSON object structured specifically for the chosen agent's /tasks/send endpoint.

Parameter Structuring Rules:

A. If "chosen_agent_name" is "curator":
   - The "parameters_for_agent" object MUST be:
     {{
       "mcp_tool_name": "[CHOOSE: 'curate_content' OR 'scrape_web_content']",
       "mcp_tool_params": {{ ...parameters for the chosen curator MCP tool... }}
     }}
   - For "curate_content", "mcp_tool_params" MUST include:
     {{
       "topic": "string",  // The main topic to curate content about
       "format": "string"  // Optional: "markdown" or "html", defaults to "markdown"
     }}
   - For "scrape_web_content", "mcp_tool_params" MUST include:
     {{
       "urls": ["string"],  // List of URLs to scrape
       "extract_type": "string"  // One of: "text", "links", "all"
     }}
   - DO NOT include any parameters not listed above.

B. If "chosen_agent_name" is "evaluator":
   - The "parameters_for_agent" object MUST be:
     {{
       "google_sheet_url": "[extracted or inferred sheet URL]",
       "task_description": "[extracted or inferred task description]"
     }}
   - Infer these parameters from the User Request.

C. If "chosen_agent_name" is null (no delegation):
   - "parameters_for_agent" should be {{}}.

Example for Curator (curate_content):
User Request: "Curate a webpage about AI in education, including examples from example.com/ai."
Response:
{{
  "chosen_agent_name": "curator",
  "parameters_for_agent": {{
    "mcp_tool_name": "curate_content",
    "mcp_tool_params": {{
      "topic": "AI in Education",
      "format": "markdown"
    }}
  }}
}}

Example for Curator (scrape_web_content):
User Request: "Scrape the text content from arxiv.org"
Response:
{{
  "chosen_agent_name": "curator",
  "parameters_for_agent": {{
    "mcp_tool_name": "scrape_web_content",
    "mcp_tool_params": {{
      "urls": ["https://arxiv.org"],
      "extract_type": "text"
    }}
  }}
}}

Example for Evaluator:
User Request: "Please evaluate the trainees based on the project 'build a REST API' using data from sheet google.com/sheet/xyz."
Response:
{{
  "chosen_agent_name": "evaluator",
  "parameters_for_agent": {{
    "google_sheet_url": "https://google.com/sheet/xyz",
    "task_description": "Evaluate based on ability to build a REST API."
  }}
}}

Example for No Delegation:
User Request: "What is the weather like today?"
Response:
{{
  "chosen_agent_name": null,
  "parameters_for_agent": {{}}
}}

Ensure your output is ONLY the JSON object, without any surrounding text or explanations.
"""
                logging.info(f"Sending delegation prompt to AI. Prompt length: {len(delegation_prompt)} chars")
                delegation_decision_response = await asyncio.to_thread(
                    instance.ai_provider.generate_content, delegation_prompt
                )
                
                delegation_decision_text = delegation_decision_response.text
                logging.info(f"AI raw response for delegation: {delegation_decision_text}")

                try:
                    json_match_delegation = re.search(r"```json\s*([\s\S]*?)\s*```", delegation_decision_text, re.IGNORECASE)
                    if json_match_delegation:
                        parsed_decision_text = json_match_delegation.group(1)
                    else:
                        parsed_decision_text = delegation_decision_text.strip()
                        first_brace = parsed_decision_text.find('{')
                        last_brace = parsed_decision_text.rfind('}')
                        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                            parsed_decision_text = parsed_decision_text[first_brace:last_brace+1]
                        
                    delegation_info = json.loads(parsed_decision_text)
                    chosen_agent_name = delegation_info.get("chosen_agent_name")
                    agent_params = delegation_info.get("parameters_for_agent")

                    if chosen_agent_name in ["curator", "evaluator"] and isinstance(agent_params, dict):
                        logging.info(f"AI chose to delegate to agent: '{chosen_agent_name}', with parameters for agent: {agent_params}")
                        forward_response = await instance._forward_to_agent(
                            agent_name=chosen_agent_name, 
                            agent_specific_payload_params=agent_params
                        )
                        
                        user_message_markdown = ""
                        if forward_response.get("success"):
                            agent_task_response = forward_response.get("agent_response")
                            if agent_task_response and isinstance(agent_task_response, dict):
                                task_status = agent_task_response.get("status", {}).get("state", "unknown_state")
                                task_id = agent_task_response.get("id", "unknown_id")
                                user_message_markdown = f"**Task `{task_id}` sent to `{chosen_agent_name}`.**\nStatus: `{task_status}`\n"
                                
                                artifacts = agent_task_response.get("artifacts")
                                if artifacts and isinstance(artifacts, list) and len(artifacts) > 0:
                                    # Process each artifact
                                    artifact_preview = ""
                                    for artifact in artifacts:
                                        if isinstance(artifact, dict):
                                            # Handle FunctionResponseContent format
                                            if "name" in artifact and "response" in artifact:
                                                actual_artifact_content = artifact.get("response", {})
                                                if isinstance(actual_artifact_content, dict):
                                                    if "generated_text_output" in actual_artifact_content:
                                                        # This is from curator's content generation
                                                        artifact_preview = actual_artifact_content["generated_text_output"]
                                                    elif "success" in actual_artifact_content:
                                                        # This is from evaluator's performance evaluation
                                                        if actual_artifact_content.get("success"):
                                                            # Get the full evaluation details
                                                            ranking_summary = actual_artifact_content.get("ranking_summary", "")
                                                            trainees_processed = actual_artifact_content.get("trainees_processed", 0)
                                                            data_sources = actual_artifact_content.get("data_sources", {})
                                                            
                                                            # Format the evaluation details
                                                            artifact_preview = f"Evaluation completed successfully.\n\n"
                                                            if ranking_summary:
                                                                artifact_preview += f"**Ranking Summary:**\n{ranking_summary}\n\n"
                                                            if trainees_processed:
                                                                artifact_preview += f"**Trainees Processed:** {trainees_processed}\n\n"
                                                            if data_sources:
                                                                artifact_preview += "**Data Sources:**\n"
                                                                if "google_sheet_url" in data_sources:
                                                                    artifact_preview += f"- Google Sheet: {data_sources['google_sheet_url']}\n"
                                                                if "gitlab_repos" in data_sources:
                                                                    artifact_preview += "- GitLab Repositories:\n"
                                                                    for repo in data_sources["gitlab_repos"]:
                                                                        artifact_preview += f"  - {repo}\n"
                                                        else:
                                                            artifact_preview = f"Evaluation failed: {actual_artifact_content.get('error', 'Unknown error')}"
                                                    elif "scraped_content" in actual_artifact_content:
                                                        # This is from curator's web scraping
                                                        scraped_data = actual_artifact_content["scraped_content"]
                                                        urls_processed = actual_artifact_content.get("urls_processed", [])
                                                        extract_type = actual_artifact_content.get("extract_type", "all")
                                                        
                                                        logging.info(f"Processing scraped content from {len(urls_processed)} URLs with type '{extract_type}'")
                                                        
                                                        # Format the scraped content in a clean way
                                                        artifact_preview = f"**Web Scraping Results**\n\n"
                                                        artifact_preview += f"Processed {len(urls_processed)} URL(s) with extraction type '{extract_type}'.\n\n"
                                                        
                                                        for url, content in scraped_data.items():
                                                            logging.info(f"Processing content from URL: {url}")
                                                            artifact_preview += f"**Content from: {url}**\n\n"
                                                            
                                                            if "text" in content:
                                                                # Use Gemini to organize the content
                                                                organization_prompt = f"""Please organize and format the following scraped web content in a clean, readable way. 
                                                                Keep only the most relevant information and maintain this exact structure:
                                                                
                                                                **Title:** [Main title or heading]
                                                                **Authors:** [If present]
                                                                **Summary:** [Brief summary or abstract]
                                                                **Key Points:** [Bullet points of main points]
                                                                **Relevant Links:** [Important links from the content]
                                                                
                                                                Here's the raw content to organize:
                                                                {content['text']}
                                                                
                                                                Respond ONLY with the formatted content following the structure above. Do not add any explanations or additional text."""
                                                                
                                                                try:
                                                                    logging.info("Sending content to Gemini for organization...")
                                                                    organized_content = await asyncio.to_thread(
                                                                        instance.ai_provider.generate_content, 
                                                                        organization_prompt
                                                                    )
                                                                    logging.info("Received response from Gemini")
                                                                    
                                                                    if organized_content and organized_content.text:
                                                                        logging.info("Using Gemini's organized content")
                                                                        # Use the organized content directly
                                                                        artifact_preview += organized_content.text + "\n\n"
                                                                    else:
                                                                        logging.warning("Empty response from Gemini, falling back to basic formatting")
                                                                        raise Exception("Empty response from Gemini")
                                                                        
                                                                except Exception as e:
                                                                    logging.error(f"Error organizing content with Gemini: {e}")
                                                                    logging.info("Falling back to basic formatting...")
                                                                    # Fallback to basic formatting if Gemini fails
                                                                    text_content = content["text"]
                                                                    # Clean up the text content
                                                                    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                                                                    text_content = re.sub(r'(Skip to main content|Help|Login|Subscribe|Copyright|Privacy Policy).*?\n', '', text_content)
                                                                    
                                                                    # Extract title and authors if present
                                                                    title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', text_content)
                                                                    authors_match = re.search(r'Authors:\s*(.*?)(?:\n|$)', text_content)
                                                                    abstract_match = re.search(r'Abstract:\s*(.*?)(?:\n\n|\nSubjects:|$)', text_content, re.DOTALL)
                                                                    
                                                                    if title_match:
                                                                        artifact_preview += f"**Title:** {title_match.group(1).strip()}\n\n"
                                                                    if authors_match:
                                                                        artifact_preview += f"**Authors:** {authors_match.group(1).strip()}\n\n"
                                                                    if abstract_match:
                                                                        artifact_preview += f"**Summary:** {abstract_match.group(1).strip()}\n\n"
                                                                    else:
                                                                        # If no structured content found, show limited raw text
                                                                        if len(text_content) > 1000:
                                                                            text_content = text_content[:1000] + "..."
                                                                        artifact_preview += f"**Text Content:**\n{text_content}\n\n"
                                                            
                                                            if "links" in content and content["links"]:
                                                                # Use Gemini to identify the most relevant links
                                                                links_prompt = f"""From the following list of links, identify the 5 most relevant ones that are likely to contain important information. 
                                                                Return ONLY a JSON array of the selected URLs:
                                                                
                                                                {json.dumps(content['links'], indent=2)}"""
                                                                
                                                                try:
                                                                    relevant_links = await asyncio.to_thread(
                                                                        instance.ai_provider.generate_content, 
                                                                        links_prompt
                                                                    )
                                                                    try:
                                                                        selected_links = json.loads(relevant_links.text)
                                                                        artifact_preview += "**Relevant Links:**\n"
                                                                        for link in selected_links:
                                                                            artifact_preview += f"- {link}\n"
                                                                        if len(content["links"]) > len(selected_links):
                                                                            artifact_preview += f"... and {len(content['links']) - len(selected_links)} more links\n"
                                                                        artifact_preview += "\n"
                                                                    except json.JSONDecodeError:
                                                                        # Fallback to showing first 5 links if JSON parsing fails
                                                                        artifact_preview += "**Relevant Links:**\n"
                                                                        for link in content["links"][:5]:
                                                                            artifact_preview += f"- {link}\n"
                                                                        if len(content["links"]) > 5:
                                                                            artifact_preview += f"... and {len(content['links']) - 5} more links\n"
                                                                        artifact_preview += "\n"
                                                                except Exception as e:
                                                                    logging.error(f"Error selecting relevant links with Gemini: {e}")
                                                                    # Fallback to showing first 5 links
                                                                    artifact_preview += "**Relevant Links:**\n"
                                                                    for link in content["links"][:5]:
                                                                        artifact_preview += f"- {link}\n"
                                                                    if len(content["links"]) > 5:
                                                                        artifact_preview += f"... and {len(content['links']) - 5} more links\n"
                                                                    artifact_preview += "\n"
                                            # Handle ErrorContent format
                                            elif "type" in artifact and artifact["type"] == "error":
                                                artifact_preview = f"Error: {artifact.get('message', 'Unknown error')}"
                                            # Handle raw text content
                                            elif "text" in artifact:
                                                artifact_preview = artifact["text"]
                                            else:
                                                artifact_preview = f"Agent '{chosen_agent_name}' provided artifact: {json.dumps(artifact, indent=2)[:500]}..."
                                        else:
                                            artifact_preview = f"Agent '{chosen_agent_name}' provided non-dict artifact: {str(artifact)[:500]}..."
                                else:
                                    artifact_preview = f"No artifacts list found in {chosen_agent_name}'s response, or list is empty."

                                if artifact_preview:
                                    user_message_markdown += f"\n\n{artifact_preview}"
                                elif agent_task_response.get("status", {}).get("message"):
                                    user_message_markdown += f" Agent message: {agent_task_response.get('status', {}).get('message')}"

                                return {
                                    "success": True,
                                    "handled_directly": False,
                                    "response": user_message_markdown,
                                    "delegation_details": forward_response
                                }
                            elif forward_response.get("agent_response_text"):
                                user_message_markdown = f"**Task sent to `{chosen_agent_name}`.**\nAgent responded with text: \n```\n{forward_response.get("agent_response_text")[:200]}...\n```"
                                return {
                                    "success": True,
                                    "handled_directly": False,
                                    "response": user_message_markdown,
                                    "delegation_details": forward_response
                                }
                            else:
                                user_message_markdown = f"**Task sent to `{chosen_agent_name}`**, but the agent's response was not in the expected format."
                                return {
                                    "success": True, 
                                    "handled_directly": False,
                                    "response": user_message_markdown,
                                    "delegation_details": forward_response
                                }
                        else: 
                            user_message_markdown = f"**Failed to delegate task to `{chosen_agent_name}`.**\nError: _{forward_response.get('error')}_"
                            if forward_response.get('agent_error_details'):
                                user_message_markdown += f"\nDetails: \n```\n{forward_response.get('agent_error_details')}\n```"
                            return {
                                "success": False,
                                "handled_directly": False,
                                "error": user_message_markdown,
                                "details": forward_response.get('agent_error_details')
                            }
                    elif chosen_agent_name is None:
                        logging.info(f"AI explicitly chose not to delegate. Request: '{request_str}'")
                        if direct_response_markdown:
                             return {
                                "success": True,
                                "handled_directly": True,
                                "response": direct_response_markdown
                            }
                        else: 
                            logging.info(f"No delegation, no prior direct answer, attempting general LLM response for: {request_str}")
                            fallback_prompt = f'Answer the following in Markdown format: "{request_str}"' 
                            fallback_response = await asyncio.to_thread(instance.ai_provider.generate_content, fallback_prompt)
                            return {
                                "success": True,
                                "handled_directly": True, 
                                "response": fallback_response.text
                            }
                    else:
                        logging.error(f"AI delegation decision was malformed. Agent: '{chosen_agent_name}', Params: {agent_params}. AI Raw: {delegation_decision_text}")
                        malformed_response_md = f"I analyzed your request but could not find a suitable specialized agent or action. Could you please rephrase or provide more details?\n\n(Debug: AI delegation response was: ```{delegation_decision_text}```)"
                        return {
                            "success": True, 
                            "handled_directly": False,
                            "response": malformed_response_md,
                            "debug_ai_delegation_response": delegation_decision_text
                        }
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse AI delegation decision as JSON: '{parsed_decision_text}'. Error: {e}")
                    error_md = f"**Error:** Failed to parse the AI's delegation plan.\nDetails: `{str(e)}`\nRaw AI Response: \n```\n{delegation_decision_text}\n```"
                    return {"success": False, "error": error_md, "details": str(e), "raw_ai_response": delegation_decision_text}
                except Exception as e:
                    logging.error(f"Error during delegation processing: {e}")
                    import traceback
                    traceback.print_exc()
                    error_md = f"**An unexpected error occurred during delegation:**\n`{str(e)}`"
                    return {"success": False, "error": error_md}

            except Exception as e:
                logging.error(f"Overall error in actual_handle_request: {e}")
                import traceback
                traceback.print_exc()
                error_md = f"**An unexpected error occurred in `actual_handle_request`:**\n`{str(e)}`"
                return {
                    "success": False,
                    "error": error_md
                }
        
        async def mcp_tool_wrapper(request: str) -> Dict[str, Any]:
            return await actual_handle_request(self, request)

        self.mcp_server.tool(name="handle_request", description="Handles a general user request, potentially delegating to specialized agents. Responds in Markdown.")(mcp_tool_wrapper)

        @self.mcp_server.tool()
        async def delegate_to_agent(agent_name: str, task: str) -> Dict[str, Any]:
            """Delegate a task to a specific agent. (Responds in Markdown-friendly format)"""
            try:
                agent_card = await self._get_agent_card(agent_name)
                if not agent_card:
                    return {
                        "success": False,
                        "error": f"**Agent `{agent_name}` not found.**"
                    }
                
                result = await self._forward_to_agent(agent_name, json.loads(task))
                
                if result.get("success"):
                    return {
                        "success": True,
                        "agent": agent_name,
                            "result_summary": f"Successfully forwarded task to `{agent_name}`. Response: ```json\n{json.dumps(result.get('agent_response', {}), indent=2)}\n```"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"**Failed to forward task to `{agent_name}`.** Details: _{result.get('error', 'Unknown error')}_",
                        "details": result.get('agent_error_details')
                    }
            except json.JSONDecodeError as e:
                 return {"success": False, "error": f"**Error decoding task parameters for `{agent_name}`.** Expected a JSON string. Error: `{str(e)}`"}
            except Exception as e:
                return {
                    "success": False,
                    "error": f"**Error delegating to `{agent_name}`:** `{str(e)}`"
                }
    
    async def _get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents from A2A server"""
        try:
            # Define known agent endpoints
            agent_endpoints = {
                "curator": "http://localhost:8001",
                "evaluator": "http://localhost:8004",
                # Add other agents as they become available
                # "trainer": "http://localhost:8003",
            }
            
            available_agents = []
            
            # Check each agent endpoint
            async with aiohttp.ClientSession() as session:
                for name, url in agent_endpoints.items():
                    try:
                        # Try to fetch agent card
                        async with session.get(f"{url}/agent.json", timeout=5) as response:
                            if response.status == 200:
                                agent_data = await response.json()
                                available_agents.append({
                                    "name": name,
                                    "url": url,
                                    "description": agent_data.get("description", ""),
                                    "capabilities": agent_data.get("capabilities", {}),
                                    "skills": agent_data.get("skills", [])
                                })
                    except Exception as e:
                        logging.error(f"Error fetching agent {name}: {str(e)}")
                        continue
            
            return available_agents
        except Exception as e:
            logging.error(f"Error getting available agents: {str(e)}")
            return []
    
    async def _get_agent_card(self, agent_name: str) -> Dict[str, Any]:
        """Get agent card for a specific agent"""
        try:
            # Get available agents
            available_agents = await self._get_available_agents()
            
            # Find the requested agent
            agent = next((a for a in available_agents if a["name"] == agent_name), None)
            
            if not agent:
                return None
            
            # Fetch full agent card
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{agent['url']}/agent.json", timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return None
        except Exception as e:
            logging.error(f"Error getting agent card for {agent_name}: {str(e)}")
            return None
    
    async def _forward_to_agent(self, agent_name: str, agent_specific_payload_params: Dict[str, Any]) -> Dict[str, Any]:
        """Forward a task to a specific agent using the /tasks/send endpoint 
           with a pre-structured payload for the agent's message content.
        """
        try:
            agent_card = await self._get_agent_card(agent_name)
            if not agent_card:
                return {"success": False, "error": f"Agent {agent_name} not found or not available"}
            
            agent_a2a_url = agent_card.get("url")
            if not agent_a2a_url:
                return {"success": False, "error": f"A2A URL not found for agent {agent_name}"}
            
            # The agent_specific_payload_params is the dictionary that will become the JSON string
            # for the "text" field of the task message content.
            task_id = f"{agent_name}-task-{str(uuid.uuid4())[:8]}" # Generate a unique task ID
            
            # Construct the payload for /tasks/send
            # agent_specific_payload_params already contains what should go into message.content.text (as a dict)
            # So we json.dumps it for the "text" field.
            payload = {
                "id": task_id,
                "message": {
                    "role": "user", # Or system, depending on how agents expect it
                    "content": {
                        "type": "text",
                        "text": json.dumps(agent_specific_payload_params)
                    }
                }
                # session_id could be added if needed: "sessionId": "your_session_id"
            }

            async with aiohttp.ClientSession() as session:
                target_url = f"{agent_a2a_url}/tasks/send"
                logging.info(f"Forwarding task to {agent_name} ({target_url}) with payload: {json.dumps(payload, indent=2)}")
                
                async with session.post(target_url, json=payload, timeout=120) as response:
                    response_text = await response.text()
                    if response.status == 200 or response.status == 201: # 201 for created
                        try:
                            response_json = json.loads(response_text)
                            # We expect the target agent to return a Task object (or its dict representation)
                            # The actual useful result from the agent's task execution would typically be in response_json.artifacts
                            # or a field within response_json.status.message if it's a simple text reply.
                            # For now, we return the whole task response from the target agent.
                            return {"success": True, "agent_response": response_json}
                        except json.JSONDecodeError:
                            logging.warning(f"WARN: Failed to parse JSON response from {agent_name} for /tasks/send. Raw: {response_text[:500]}")
                            # Even if not JSON, if it was a 2xx, it might be a simple text success message
                            return {"success": True, "agent_response_text": response_text}
                    else:
                        logging.error(f"ERROR from {agent_name} ({target_url}): HTTP {response.status} - {response_text[:500]}")
                        return {
                            "success": False,
                            "error": f"Error from {agent_name}: HTTP {response.status}",
                            "agent_error_details": response_text[:500]
                        }
        except Exception as e:
            import traceback
            logging.error(f"Unexpected error in _forward_to_agent for {agent_name}: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": f"An unexpected error occurred while forwarding task to {agent_name}: {str(e)}"
            }
    
    async def handle_a2a_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle A2A task requests"""
        skill = task_data.get("skill")
        params = task_data.get("params", {})
        
        logging.info(f"Handling A2A task - Skill: {skill}, Params: {params}")
        
        if skill == "general_conversation":
            return await self.mcp_server.call_tool("handle_request", **params)
        elif skill == "task_coordination":
            return await self.mcp_server.call_tool("delegate_to_agent", **params)
        else:
            return {"success": False, "error": f"Unknown skill: {skill}"}
    
    async def start_servers(self, a2a_port: int = 3000, mcp_port: int = 3001):
        """Start both A2A and MCP servers"""
        logging.info(f"Starting Coordinator Agent servers...")
        logging.info(f"A2A Server: http://localhost:{a2a_port}")
        logging.info(f"MCP Server: http://localhost:{mcp_port}")
        
        # Create Flask app for A2A server and chat interface
        a2a_app = Flask(__name__, template_folder='templates')
        self.a2a_server = A2AServer(
            agent_card=self.agent_card,
            task_handler=self.handle_a2a_task,
            port=a2a_port
        )
        self.a2a_server.setup_routes(a2a_app)
        
        # Add route for the chat interface HTML
        @a2a_app.route('/chat_ui')
        def chat_ui():
            return send_from_directory('templates', 'chat.html')

        # Add route to handle chat messages
        @a2a_app.route('/chat', methods=['POST'])
        def handle_chat():
            data = flask_request.get_json()
            user_message = data.get('message')
            if not user_message:
                return jsonify({'error': 'No message provided'}), 400

            mcp_response_obj = None
            actual_response_dict = None
            try:
                # Run the async MCP tool call in a new event loop
                mcp_response_obj = asyncio.run(self.mcp_server.call_tool(tool_name="handle_request", params={"request": user_message}))
                
                if mcp_response_obj:
                    if not mcp_response_obj.is_error:
                        if mcp_response_obj.content and isinstance(mcp_response_obj.content, list) and len(mcp_response_obj.content) > 0:
                            content_item = mcp_response_obj.content[0]
                            if isinstance(content_item, dict) and content_item.get("type") == "text" and "text" in content_item:
                                try:
                                    actual_response_dict = json.loads(content_item["text"])
                                except json.JSONDecodeError as jde:
                                    logging.error(f"ERROR: Failed to parse JSON from MCPResponse content: {jde}. Content: {content_item['text']}")
                                    # If it's not JSON, it might be Markdown directly from a simple tool response
                                    actual_response_dict = {"success": True, "response": content_item["text"]}
                            else:
                                logging.warning(f"WARN: MCPResponse content item is not in the expected format: {content_item}")
                                actual_response_dict = {"success": False, "error": "Agent response content in unexpected format."}
                        else:
                            logging.warning(f"WARN: MCPResponse content is empty or not a list: {mcp_response_obj.content}")
                            actual_response_dict = {"success": False, "error": "Agent provided no parsable content."}
                    else: # mcp_response_obj.is_error is True
                        error_text = "Unknown error from MCP tool."
                        if mcp_response_obj.content and isinstance(mcp_response_obj.content, list) and len(mcp_response_obj.content) > 0:
                            content_item = mcp_response_obj.content[0]
                            if isinstance(content_item, dict) and "text" in content_item:
                                error_text = content_item["text"]
                        logging.error(f"ERROR: MCP tool call resulted in an MCPError: {error_text}")
                        actual_response_dict = {"success": False, "error": f"Agent Error: {error_text}"}
                else:
                    logging.error("ERROR: MCP tool call returned None")
                    actual_response_dict = {"success": False, "error": "No response from agent processing."}

            except RuntimeError as e:
                logging.error(f"ERROR: asyncio.run() in handle_chat failed: {e}.")
                return jsonify({'reply': f"Sorry, I encountered an internal issue processing your request. (Async Error: {e})"}), 500
            except Exception as e:
                logging.error(f"ERROR: General exception in handle_chat calling MCP tool: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'reply': f"Sorry, an unexpected error occurred. ({e})"}), 500

            agent_reply = "Sorry, I couldn't process your request properly. Please check the logs for details."

            if actual_response_dict:
                if actual_response_dict.get('success'):
                    agent_reply = actual_response_dict.get('response', "Request processed, but no specific Markdown reply content found.")
                elif 'error' in actual_response_dict:
                    agent_reply = actual_response_dict.get('error', "An unknown error occurred.") # error should also be Markdown
            
            if not isinstance(agent_reply, str):
                logging.warning(f"WARN: agent_reply is not a string, converting. Type: {type(agent_reply)}, Value: {agent_reply}")
                agent_reply = str(agent_reply)

            return jsonify({'reply': agent_reply})
        
        # Create FastAPI app for MCP server
        mcp_app = create_fastapi_app(self.mcp_server)
        
        # Start both servers concurrently
        import uvicorn
        await asyncio.gather(
            asyncio.to_thread(lambda: a2a_app.run(host='0.0.0.0', port=a2a_port)),
            asyncio.to_thread(lambda: uvicorn.run(mcp_app, host='0.0.0.0', port=mcp_port))
        )

if __name__ == "__main__":
    coordinator = CoordinatorAgent()
    asyncio.run(coordinator.start_servers()) 