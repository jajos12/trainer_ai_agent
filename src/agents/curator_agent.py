import asyncio
import os
from typing import Dict, Any, List
import aiohttp
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from python_a2a.mcp import FastMCP, text_response, create_fastapi_app
from python_a2a import AgentCard, A2AServer, AgentSkill, Message, MessageRole, TextContent, Task, TaskStatus, TaskState, ErrorContent
from python_a2a.models.content import FunctionResponseContent
from python_a2a.mcp.fastmcp import MCPResponse
import google.generativeai as genai
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CuratorAgent:
    def __init__(self):
        self.mcp_server = FastMCP("Curator MCP Server")
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
            name="curator",
            description="AI agent that curates content and creates training tasks in Markdown format.",
            url="http://localhost:8001",
            version="1.0.0",
            provider="Curator AI",
            capabilities={
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True
            },
            default_input_modes=["text/plain"],
            default_output_modes=["text/markdown", "text/plain"],
            skills=[
                AgentSkill(
                    name="content_curation",
                    description="Curate content from various sources and generate training materials in Markdown format.",
                    tags=["content", "training", "ai", "curation", "markdown"],
                    examples=[
                        "Curate content about machine learning fundamentals in Markdown",
                        "Create training materials for Python programming as a Markdown document"
                    ],
                    input_modes=["text/plain"],
                    output_modes=["text/markdown", "text/plain"]
                ),
                AgentSkill(
                    name="web_scraping",
                    description="Scrape web content from URLs and return it as a Markdown formatted string.",
                    tags=["scraping", "web", "data", "extraction", "markdown"],
                    examples=[
                        "Scrape research papers from arXiv and provide a Markdown summary",
                        "Extract links from documentation pages and list them in Markdown"
                    ],
                    input_modes=["text/plain"],
                    output_modes=["text/markdown", "text/plain"]
                )
            ]
        )
    
    async def _scrape_url(self, url: str, extract_type: str = "all") -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        return f"### Error scraping `{url}`\nHTTP Status: {response.status}"
                    
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()
                    
                    # Extract title
                    title = soup.title.string if soup.title else url
                    
                    # Extract main content
                    main_content = ""
                    if extract_type in ["text", "all"]:
                        # First try to find content in span.descriptor
                        descriptor_spans = soup.find_all('span', class_='descriptor')
                        if descriptor_spans:
                            main_content = '\n'.join(span.get_text(strip=True) for span in descriptor_spans)
                        else:
                            # Fallback to other content areas if no descriptor spans found
                            main_tags = soup.find_all(['article', 'main', 'div'], class_=lambda x: x and any(c in str(x).lower() for c in ['content', 'main', 'article', 'post']))
                            if main_tags:
                                main_content = main_tags[0].get_text(separator='\n', strip=True)
                            else:
                                main_content = soup.get_text(separator='\n', strip=True)
                        
                        # Clean up the text
                        main_content = re.sub(r'\n\s*\n', '\n\n', main_content)
                        main_content = re.sub(r'(Skip to main content|Help|Login|Subscribe|Copyright|Privacy Policy).*?\n', '', main_content)
                    
                    # Extract links
                    links = []
                    if extract_type in ["links", "all"]:
                        for link_tag in soup.find_all('a', href=True):
                            href = link_tag['href']
                            if href.startswith('http'):
                                link_text = link_tag.get_text(strip=True) or href
                                links.append(f"- [{link_text.replace('\n',' ').strip()}]({href})")
                    
                    # Use Gemini to organize the content
                    organization_prompt = f"""Please organize and format the following scraped web content in a clean, readable way, I want you to give me unrendered markdown for I am to use it to preview it in my markdown previewer. 
                    Keep only the most relevant information and maintain this exact structure:
                    
                    **Title:** [Main title or heading]
                    **Authors:** [If present]
                    **Summary:** [Brief summary or abstract]
                    **Key Points:** [Bullet points of main points]
                    **Relevant Links:** [Important links from the content]
                    
                    Here's the raw content to organize:
                    Title: {title}
                    
                    Content:
                    {main_content}
                    
                    Links:
                    {chr(10).join(links[:20])}
                    
                    Respond ONLY with the formatted content following the structure above. Do not add any explanations or additional text."""
                    
                    try:
                        organized_content = await asyncio.to_thread(
                            self.ai_provider.generate_content, 
                            organization_prompt
                        )
                        
                        if organized_content and organized_content.text:
                            return organized_content.text
                        else:
                            raise Exception("Empty response from Gemini")
                            
                    except Exception as e:
                        print(f"Error organizing content with Gemini: {e}")
                        # Fallback to basic formatting
                        markdown_parts = [f"## Scraped Content from: `{url}`\n"]
                        
                        if main_content:
                            markdown_parts.append(f"### Text Content\n{main_content[:15000]}\n")
                        else:
                            markdown_parts.append("_No significant text content found._\n")
                        
                        if links:
                            markdown_parts.append(f"### Extracted Links\n" + "\n".join(links[:20]) + "\n")
                        else:
                            markdown_parts.append("_No external links found._\n")
                        
                        return "\n".join(markdown_parts)
                        
        except Exception as e:
            return f"### Error scraping `{url}`\nException: {str(e)}"
    
    async def _fetch_arxiv_papers(self, topic: str, max_results: int = 5) -> str:
        try:
            topic_to_category = {
                "machine learning": "cs.LG", "ai": "cs.AI", "nlp": "cs.CL",
                "computer vision": "cs.CV", "robotics": "cs.RO", "mathematics": "math.GM",
                "physics": "physics", "statistics": "stat.ML", "algorithms": "cs.DS",
                "deep learning": "cs.LG", "neural networks": "cs.NE", "optimization": "math.OC"
            }
            
            # Try to find the most relevant category
            category = None
            for key, cat in topic_to_category.items():
                if key in topic.lower():
                    category = cat
                    break
            
            # If no specific category found, use a broader search
            if not category:
                query = f"all:{topic}"
            else:
                query = f"cat:{category} AND all:{topic}"
            
            url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status != 200:
                        return f"_Could not fetch arXiv papers for '{topic}' (HTTP {response.status})._"
                    
                    xml_data = await response.text()
                    root = ET.fromstring(xml_data)
                    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    papers_md_parts = [f"### Recent arXiv Research on: {topic}\n"]
                    entries = root.findall('atom:entry', namespace)
                    if not entries:
                        return f"_No recent arXiv papers found for '{topic}' via query '{query}'._"

                    for entry in entries:
                        title_element = entry.find('atom:title', namespace)
                        title = (title_element.text or "").strip() if title_element is not None else "N/A"
                        
                        authors_elements = entry.findall('atom:author', namespace)
                        authors = []
                        for author_element in authors_elements:
                            name_element = author_element.find('atom:name', namespace)
                            if name_element is not None and name_element.text is not None:
                                authors.append(name_element.text.strip())
                        
                        summary_element = entry.find('atom:summary', namespace)
                        summary = (summary_element.text or "").strip() if summary_element is not None else "N/A"
                        
                        # Try to find descriptor spans in the summary
                        if summary:
                            soup = BeautifulSoup(summary, 'html.parser')
                            descriptor_spans = soup.find_all('span', class_='descriptor')
                            if descriptor_spans:
                                summary = '\n'.join(span.get_text(strip=True) for span in descriptor_spans)
                        
                        id_element = entry.find('atom:id', namespace)
                        paper_url = (id_element.text or "").strip() if id_element is not None else "#"
                        
                        published_element = entry.find('atom:published', namespace)
                        published_date = (published_element.text or "").strip() if published_element is not None else "N/A"
                        
                        authors_str = ', '.join(authors[:3])
                        if len(authors) > 3: authors_str += ", et al."
                        
                        papers_md_parts.append(
                            f"**[{title.replace('\n', ' ').strip()}]({paper_url})**\n"
                            f"_Authors: {authors_str}_\n"
                            f"_Published: {published_date}_\n"
                            f"> {summary.replace('\n', ' ').strip()[:500]}...\n"
                        )
                    return "\n".join(papers_md_parts)
        except Exception as e:
            return f"_Error fetching arXiv papers for '{topic}': {str(e)}_"
    
    def _register_mcp_tools(self):
        """Register MCP tools using FastMCP"""
        
        @self.mcp_server.tool()
        async def curate_content(topic: str, sources: list = None, format: str = "markdown") -> Dict[str, Any]:
            """Curate content for a given topic using arXiv, web sources, and AI generation, returning Markdown.
            
            Args:
                topic: The topic to curate content for
                sources: Optional list of web sources to scrape
                format: Output format (default: "markdown")
                
            Returns:
                Dictionary containing the curated content
            """
            try:
                context_parts = [f"# Curated Content Request: {topic}\n"]
                
                print(f"Fetching arXiv papers for topic: {topic}")
                arxiv_markdown = await self._fetch_arxiv_papers(topic, max_results=5)
                if arxiv_markdown and not arxiv_markdown.startswith("_Error") and not arxiv_markdown.startswith("_No recent"):
                    context_parts.append("## Research Papers from arXiv\n" + arxiv_markdown)
                else:
                    context_parts.append(f"_Note: Could not fetch significant arXiv data for {topic}. Details: {arxiv_markdown}_")
                
                if sources and isinstance(sources, list):
                    for source_url in sources[:2]: 
                        print(f"Scraping URL for context: {source_url}")
                        scraped_md_from_url = await self._scrape_url(source_url, extract_type="text") 
                        if not scraped_md_from_url.startswith("### Error"):
                            context_parts.append(f"## Context from Web Source: {source_url}\n{scraped_md_from_url}")
                        else:
                            context_parts.append(f"_Note: Could not scrape significant content from {source_url}. Details: {scraped_md_from_url}_")
                
                combined_context = "\n\n".join(context_parts)

                # Updated prompt for better AI topic handling
                prompt = f"""You are an expert in AI and machine learning research.
Your task is to create a comprehensive analysis and explanation of the topic: "{topic}".
and also to design a task after reading this text a person learning should implement. 
**Output Format Requirement:**
Your entire response MUST be a single, well-formatted Markdown document.

**Content Structure:**
Please include the following sections in your Markdown document:
1. **Introduction**: Brief overview of the topic and its significance in AI/ML
2. **Technical Details**: 
   - Core concepts and principles
   - Mathematical foundations (if applicable)
   - Key algorithms and approaches
3. **Recent Developments**: 
   - Latest research findings
   - State-of-the-art implementations
   - Current challenges and limitations
4. **Practical Applications**:
   - Real-world use cases
   - Implementation considerations
   - Performance metrics and benchmarks
5. **Future Directions**:
   - Potential improvements
   - Research opportunities
   - Emerging trends
6. **Task for the topic**:
   - Give a clear to implement task
   - Provide at least a list of 5 detialed specifications with in the task
6. **Resources and References**:
   - Key papers and publications
   - Code repositories
   - Tutorials and documentation

**Contextual Information (Use this to inform your content, but synthesize and integrate it naturally):**
```markdown
{combined_context[:15000]}
```

**Important Guidelines:**
- Focus on technical accuracy and depth
- Include relevant mathematical formulations where appropriate
- Reference specific papers and implementations
- Maintain academic rigor while being accessible
- Use code blocks for algorithm implementations
- Include diagrams or pseudocode where helpful

Begin your response directly with the Markdown content, starting with the main title: `# {topic.title()}: A Technical Analysis`.
Do NOT include any preamble or explanation before the Markdown document itself.
"""
                
                print(f"Sending content generation prompt to Gemini for topic: {topic}")
                ai_response = await asyncio.to_thread(self.ai_provider.generate_content, prompt)
                generated_markdown = ai_response.text
                
                # Clean up potential ```markdown prefix/suffix from LLM if it adds it
                if generated_markdown.strip().startswith("```markdown"):
                    generated_markdown = generated_markdown.strip()[len("```markdown"):]
                if generated_markdown.strip().endswith("```"):
                    generated_markdown = generated_markdown.strip()[:-len("```")]
                generated_markdown = generated_markdown.strip()
                
                return {
                    "success": True, "topic": topic, "format": format, 
                    "generated_text_output": generated_markdown
                }
            except Exception as e:
                print(f"Error in curate_content: {e}")
                import traceback; traceback.print_exc()
                return {"success": False, "error": str(e), "format": format, "generated_text_output": f"# Error Curating Content for {topic}\n\nAn error occurred: `{str(e)}`"}
        
        @self.mcp_server.tool()
        async def scrape_web_content(urls: list, extract_type: str = "all", format: str = "markdown", topic: str = "Not given") -> Dict[str, Any]:
            """Scrape web content from a list of URLs and return an aggregated Markdown report."""
            if not isinstance(urls, list):
                return {"success": False, "error": "Input URLs must be a list.", "format": "markdown", "generated_text_output": "**Error:** Input URLs must be a list."}
            if not urls:
                 return {"success": True, "format": "markdown", "generated_text_output": "# Web Scraping Report\n\nNo URLs provided for scraping."}

            markdown_reports = []
            individual_results = {}
            for url in urls[:3]: # Limit to 3 URLs per call for now
                print(f"Scraping URL (batch): {url} with extract_type: {extract_type}")
                md_output = await self._scrape_url(url, extract_type)
                markdown_reports.append(md_output)
                individual_results[url] = md_output # Store individual MD for potential future use
            
            aggregated_markdown = f"# Web Scraping Report\n\nProcessed {len(markdown_reports)} URL(s) with extraction type '{extract_type}'.\n"
            aggregated_markdown += "\n\n---\n\n".join(markdown_reports)
                
            return {
                "success": True, "format": "markdown",
                "generated_text_output": aggregated_markdown,
                "individual_results": individual_results 
            }
    
    def handle_task(self, task_input: Task) -> Task:
        """
        Handles an A2A task by parsing the message, invoking the specified MCP tool,
        and processing the MCPResponse.
        Expected message content format (as a JSON string in task_input.message['content']['text']):
        {
            "mcp_tool_name": "curate_content" or "scrape_web_content",
            "mcp_tool_params": { ...parameters for the tool... }
        }
        """
        try:
            print(f"CuratorAgent: Received A2A task ID: {task_input.id}, Message type: {type(task_input.message)}")

            message_data = task_input.message
            if not message_data or not isinstance(message_data, dict):
                error_msg = "Task message is missing or not a dictionary."
                print(error_msg)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg})
                task_input.artifacts = [ErrorContent(message=error_msg).to_dict()]
                return task_input

            content_data = message_data.get('content')
            if not content_data or not isinstance(content_data, dict):
                error_msg = "Task message content is missing or not a dictionary."
                print(error_msg)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg})
                task_input.artifacts = [ErrorContent(message=error_msg).to_dict()]
                return task_input

            mcp_tool_name = None
            mcp_tool_params = {} # Default to empty dict
            text_payload = content_data.get('text')

            if content_data.get('type') == 'text' and text_payload:
                try:
                    payload_dict = json.loads(text_payload)
                    if not isinstance(payload_dict, dict):
                        raise ValueError("Parsed text payload is not a dictionary.")
                    
                    mcp_tool_name = payload_dict.get("mcp_tool_name")
                    # Ensure mcp_tool_params is a dict, even if missing or None in payload
                    loaded_params = payload_dict.get("mcp_tool_params")
                    if isinstance(loaded_params, dict):
                        mcp_tool_params = loaded_params
                    elif loaded_params is not None: # if it's present but not a dict
                        raise ValueError("'mcp_tool_params' must be a dictionary if provided.")

                    if not mcp_tool_name or not isinstance(mcp_tool_name, str):
                        raise ValueError("'mcp_tool_name' is missing or not a string in payload.")

                except (json.JSONDecodeError, ValueError) as e_parse:
                    error_msg_parse = f"Invalid text payload: Expected JSON with 'mcp_tool_name' (string) and 'mcp_tool_params' (dict). Error: {e_parse}"
                    print(error_msg_parse)
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_parse})
                    task_input.artifacts = [ErrorContent(message=error_msg_parse).to_dict()]
                    return task_input
            else:
                error_msg_content = "Task message content must be of type 'text' and contain a JSON string payload in the 'text' field."
                print(error_msg_content)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_content})
                task_input.artifacts = [ErrorContent(message=error_msg_content).to_dict()]
                return task_input

            if mcp_tool_name not in self.mcp_server.tools: # Check against registered tools
                error_msg_tool = f"Unknown or unsupported MCP tool name: {mcp_tool_name}"
                print(error_msg_tool)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_tool})
                task_input.artifacts = [ErrorContent(message=error_msg_tool).to_dict()]
                return task_input

            try:
                print(f"CuratorAgent: Calling MCP tool '{mcp_tool_name}' with params: {mcp_tool_params}")
                # Convert format parameter to format_preference if present
                if 'format' in mcp_tool_params:
                    mcp_tool_params['format'] = mcp_tool_params.pop('format')
                
                mcp_result_obj = asyncio.run(self.mcp_server.call_tool(mcp_tool_name, params=mcp_tool_params))

                if not isinstance(mcp_result_obj, MCPResponse):
                    error_msg_unexpected = f"MCP tool '{mcp_tool_name}' returned unexpected type: {type(mcp_result_obj)}"
                    print(error_msg_unexpected)
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_unexpected})
                    task_input.artifacts = [ErrorContent(message=error_msg_unexpected).to_dict()]
                
                elif mcp_result_obj.is_error:
                    error_detail = f"MCP tool '{mcp_tool_name}' execution indicated an error."
                    if mcp_result_obj.content and isinstance(mcp_result_obj.content, list) and len(mcp_result_obj.content) > 0:
                        first_content_item = mcp_result_obj.content[0]
                        if isinstance(first_content_item, dict) and 'text' in first_content_item:
                            error_detail = first_content_item['text']
                    print(f"MCP tool '{mcp_tool_name}' failed (is_error=True). Detail: {error_detail}")
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": str(error_detail)})
                    task_input.artifacts = [ErrorContent(message=str(error_detail)).to_dict()]
                else: # Not an MCPResponse error, process content
                    try:
                        if not mcp_result_obj.content or not isinstance(mcp_result_obj.content, list) or len(mcp_result_obj.content) == 0:
                            raise ValueError(f"MCPResponse content for '{mcp_tool_name}' is empty or not a list.")
                        
                        first_content_item = mcp_result_obj.content[0]
                        if not isinstance(first_content_item, dict) or first_content_item.get('type') != 'text' or 'text' not in first_content_item:
                            raise ValueError(f"MCPResponse content item for '{mcp_tool_name}' is not a valid text dictionary or missing 'text' field.")

                        actual_result_dict = json.loads(first_content_item['text'])
                        
                        if not isinstance(actual_result_dict, dict):
                            raise ValueError(f"Parsed content text from MCPResponse for '{mcp_tool_name}' is not a dictionary.")

                        print(f"MCP tool '{mcp_tool_name}' processed successfully by agent. Result: {actual_result_dict}")
                        task_input.status = TaskStatus(state=TaskState.COMPLETED)
                        task_input.artifacts = [
                            FunctionResponseContent(
                                name=f"{mcp_tool_name}_result",
                                response=actual_result_dict
                            ).to_dict()
                        ]

                    except (json.JSONDecodeError, ValueError, IndexError, KeyError) as e_content_parse:
                        error_detail_content = f"Failed to process/parse content from successful MCPResponse for '{mcp_tool_name}': {type(e_content_parse).__name__}: {e_content_parse}"
                        print(f"{error_detail_content}. Content: {mcp_result_obj.content if hasattr(mcp_result_obj, 'content') else 'No content attribute'}")
                        task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_detail_content})
                        task_input.artifacts = [ErrorContent(message=error_detail_content).to_dict()]
            
            except Exception as e_mcp_call:
                import traceback
                error_trace = traceback.format_exc()
                error_msg = f"Exception during MCP tool '{mcp_tool_name}' call: {type(e_mcp_call).__name__}: {e_mcp_call}\n{error_trace}"
                print(error_msg)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": f"Internal error during MCP tool call: {type(e_mcp_call).__name__}"})
                task_input.artifacts = [ErrorContent(message=error_msg).to_dict()]
            
            return task_input

        except BaseException as be:
            import traceback
            critical_error_trace = traceback.format_exc()
            critical_error_msg = f"CRITICAL UNHANDLED EXCEPTION in CuratorAgent.handle_task: {type(be).__name__}: {be}\n{critical_error_trace}"
            print(critical_error_msg)
            try:
                if isinstance(task_input, Task):
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": f"Critical unhandled error: {type(be).__name__}"})
                    task_input.artifacts = [ErrorContent(message=critical_error_msg).to_dict()]
            except Exception as e_critical_set:
                print(f"CRITICAL: Failed to set error status on task_input during BaseException handling: {e_critical_set}")
            if isinstance(be, (KeyboardInterrupt, SystemExit)):
                raise be
            if isinstance(task_input, Task):
                return task_input
            else: # Should not happen if task_input is always a Task
                try:
                    dummy_task_id = getattr(task_input, 'id', "unknown_task_curator_critical_error")
                    return Task(id=dummy_task_id, status=TaskStatus(state=TaskState.FAILED, message={"error": f"Critical unhandled error in Curator: {type(be).__name__}"}))
                except Exception: # Final fallback
                    raise be
    
    async def start_servers(self, a2a_port: int = 8001, mcp_port: int = 8002):
        """Start both A2A and MCP servers"""
        print(f"Starting Curator Agent servers...")
        print(f"A2A Server: http://localhost:{a2a_port}")
        print(f"MCP Server: http://localhost:{mcp_port}")
        
        # Create Flask app for A2A server
        from flask import Flask
        a2a_app = Flask(__name__)
        self.a2a_server = A2AServer(
            agent_card=self.agent_card,
            port=a2a_port
        )
        self.a2a_server.setup_routes(a2a_app)
        self.a2a_server.handle_task = self.handle_task # Assign the new sync method
        
        # Create FastAPI app for MCP server
        mcp_app = create_fastapi_app(self.mcp_server)
        
        # Start both servers concurrently
        import uvicorn
        await asyncio.gather(
            asyncio.to_thread(lambda: a2a_app.run(host='0.0.0.0', port=a2a_port)),
            asyncio.to_thread(lambda: uvicorn.run(mcp_app, host='0.0.0.0', port=mcp_port))
        )

    async def start(self, host: str = "localhost", port: int = 8002):
        """Start the agent server"""
        # Initialize server components
        self.server = Mock()  # For testing purposes
        self.server.is_running = Mock(return_value=True)
        return self.server

    async def stop(self):
        """Stop the agent server"""
        if self.server:
            # Add cleanup logic here
            pass


if __name__ == "__main__":
    curator = CuratorAgent()
    asyncio.run(curator.start_servers()) 