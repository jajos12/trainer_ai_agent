# src/agents/evaluator_agent.py
import asyncio
import os
from typing import Dict, Any, List
import aiohttp
import pandas as pd # For handling Excel data
from io import BytesIO # For handling file-like data from HTTP requests

# Assuming python-a2a and genai are in the environment
from python_a2a.mcp import FastMCP, create_fastapi_app
from python_a2a import AgentCard, A2AServer, AgentSkill, Message, MessageRole, TextContent, Task, TaskStatus, TaskState, ErrorContent
from python_a2a.models.content import FunctionResponseContent # Ensure this is imported
from python_a2a.mcp.fastmcp import MCPResponse # IMPORT MCPResponse
import google.generativeai as genai

import gitlab
# Placeholder for GitLab API interaction library (e.g., python-gitlab)
# import gitlab
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EvaluatorAgent():
    def __init__(self):
        self.mcp_server = FastMCP("Evaluator MCP Server")
        self.ai_provider = self._setup_gemini()
        self.agent_card = self._create_agent_card()
        self.a2a_server = None
        self.gitlab_client = self._setup_gitlab_client() # Initialize GitLab client
        
        self._register_mcp_tools()
    
    def _setup_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    
    def _create_agent_card(self) -> AgentCard:
        return AgentCard(
            name="evaluator",
            description="AI agent that evaluates trainer performance based on GitLab repositories and task descriptions.",
            url="http://localhost:8004", # Assuming a new port for this agent
            version="1.0.0",
            provider="Evaluator AI",
            capabilities={"streaming": True, "stateTransitionHistory": True},
            default_input_modes=["application/json"],
            default_output_modes=["application/json", "text/markdown"],
            skills=[
                AgentSkill(
                    name="performance_evaluation",
                    description="Evaluates trainer performance by analyzing GitLab repositories against a task description, using data from a Google Sheet.",
                    tags=["evaluation", "gitlab", "ranking", "performance", "excel"],
                    examples=[
                        "Evaluate trainers based on the task 'build a REST API' using trainee data from [Google Sheet Link]",
                    ],
                    input_modes=["application/json"], # Expecting JSON with sheet_url and task_description
                    output_modes=["application/json", "text/markdown"]
                )
            ]
        )

    async def _fetch_google_sheet_data(self, sheet_url: str) -> List[Dict[str, str]]:
        """
        Fetches and parses data from a Google Sheet URL.
        The sheet should be publicly accessible or the agent needs appropriate permissions.
        Expected columns: 'Trainer Name', 'GitLab Repo URL' (or similar).
        """
        # This is a simplified placeholder. Real implementation requires Google Sheets API.
        # For Google Sheets, the URL needs to be modified for direct CSV export.
        # Example: replace '/edit#gid=' with '/export?format=csv&gid='
        if "/edit" in sheet_url:
            export_url = sheet_url.replace("/edit", "/export?format=csv&gid=")
            if "#gid=" in export_url: # Handle common Google Sheet URL format
                 export_url = export_url.split("#gid=")[0] + "&gid=" + sheet_url.split("#gid=")[1] if len(sheet_url.split("#gid=")) > 1 else export_url.replace("/export?format=csv&gid=", "/export?format=csv") # if no gid, export first sheet
            else: # if /edit but no #gid= present
                export_url = sheet_url.split("/edit")[0] + "/export?format=csv"

        else: # If it's already some kind of direct link (less likely for Google Docs)
            export_url = sheet_url

        print(f"Attempting to fetch Google Sheet data from: {export_url}")
        trainees = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(export_url, timeout=20) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Use BytesIO to treat the byte content as a file
                        df = pd.read_csv(BytesIO(content))
                        # Assuming columns are 'Trainer Name' and 'GitLab Repo URL'
                        # Adjust column names as necessary based on the actual sheet
                        required_cols = [col for col in df.columns if "name" in col.lower() or "repo" in col.lower()]
                        if len(required_cols) < 2:
                            raise ValueError(f"Sheet must contain columns for trainer names and repo URLs. Found: {df.columns.tolist()}")

                        name_col = next((col for col in df.columns if "name" in col.lower()), None)
                        repo_col = next((col for col in df.columns if "repo" in col.lower() or "url" in col.lower()), None)

                        if not name_col or not repo_col:
                             raise ValueError(f"Could not reliably identify Name and Repo URL columns. Found: {df.columns.tolist()}")


                        for _, row in df.iterrows():
                            trainees.append({
                                "name": str(row[name_col]),
                                "repo_url": str(row[repo_col])
                            })
                        return trainees
                    else:
                        error_text = await response.text()
                        print(f"Error fetching Google Sheet: HTTP {response.status} - {error_text[:200]}")
                        raise A2AIOException(f"Failed to fetch Google Sheet: HTTP {response.status}. Ensure the link is a direct CSV export link and publicly accessible.")
        except Exception as e:
            print(f"Exception while fetching Google Sheet: {e}")
            # In a real scenario, provide more specific error handling
            raise A2AIOException(f"Could not process Google Sheet URL: {e}")

    def _setup_gitlab_client(self):
        gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com")
        private_token = os.getenv("GITLAB_PRIVATE_TOKEN")
        
        if not private_token:
            print("Warning: GITLAB_PRIVATE_TOKEN environment variable not set. GitLab API access might be limited.")
            return None # Or initialize for anonymous access if library supports and it's desired

        try:
            gl = gitlab.Gitlab(gitlab_url, private_token=private_token)
            gl.auth() # Verify authentication by trying to access user info
            print("Successfully authenticated with GitLab API.")
            return gl
        except Exception as e:
            print(f"Error initializing GitLab client: {e}. Check GITLAB_URL and GITLAB_PRIVATE_TOKEN.")
            return None

    async def _fetch_gitlab_repo_content(self, repo_url: str) -> Dict[str, Any]:
        """
        Fetches README.md and commit messages from a GitLab repository using the GitLab API.
        """
        if not self.gitlab_client:
            print("GitLab client not initialized. Check GITLAB_PRIVATE_TOKEN. Trying to     ")
            self._setup_gitlab_client()
            

        print(f"Fetching content for GitLab repo: {repo_url}")
        
        # Basic parsing of repo_url to get project_path.
        # Example: "https://gitlab.com/username/projectname" -> "username/projectname"
        # This might need to be more robust depending on URL variations.
        try:
            if "://" in repo_url:
                project_path = "/".join(repo_url.split("/")[3:])
            else: # Assuming format like "username/projectname" if no schema
                project_path = repo_url
            
            # Remove .git if present
            if project_path.endswith(".git"):
                project_path = project_path[:-4]

            project = await asyncio.to_thread(self.gitlab_client.projects.get, project_path)
        except Exception as e:
            print(f"Error getting GitLab project '{project_path}' from URL '{repo_url}': {e}")
            return {"error": f"Could not access GitLab project: {e}", "readme": None, "commits": []}

        content_summary = {"url": repo_url, "readme": None, "commits": []}

        # Fetch README
        try:
            # Attempt to get the default README file name (e.g. README.md)
            # The actual README filename might vary (README.rst, etc.)
            # For simplicity, we assume README.md. A more robust solution would list files.
            readme_file_name = project.attributes.get('readme_url')
            if readme_file_name: # readme_url is a full URL to the file in default branch
                readme_file_name = readme_file_name.split('/')[-1] # extract filename
            else: # fallback if readme_url is not in attributes (might happen)
                readme_file_name = "README.md"
            
            # Get the default branch if not specified, or use 'main'/'master' as fallbacks
            default_branch = project.attributes.get('default_branch', 'main')

            readme_content_bytes = await asyncio.to_thread(
                project.files.get(file_path=readme_file_name, ref=default_branch).decode
            )
            content_summary["readme"] = readme_content_bytes.decode('utf-8')[:2000] # Limit size
            print(f"Successfully fetched {readme_file_name} from {project.web_url}")
        except gitlab.exceptions.GitlabGetError as e:
            if e.response_code == 404:
                print(f"README.md not found in project {project_path} on branch {default_branch}.")
            else:
                print(f"Error fetching README for {project_path}: {e}")
            content_summary["readme"] = "README file not found or error fetching."
        except Exception as e:
            print(f"Generic error fetching README for {project_path}: {e}")
            content_summary["readme"] = f"Error fetching README: {e}"


        # Fetch commits (e.g., last 5 commits)
        try:
            commits_data = await asyncio.to_thread(project.commits.list, get_all=False, per_page=5)
            for commit in commits_data:
                content_summary["commits"].append({
                    "id": commit.id,
                    "title": commit.title,
                    "message": commit.message.strip(),
                    "author_name": commit.author_name,
                    "authored_date": commit.authored_date
                })
            print(f"Successfully fetched {len(content_summary['commits'])} commits for {project.web_url}")
        except Exception as e:
            print(f"Error fetching commits for {project_path}: {e}")
            content_summary["commits"] = [{"error": f"Could not fetch commits: {e}"}]
            
        return content_summary

    def _register_mcp_tools(self):
        @self.mcp_server.tool()
        async def evaluate_performance(google_sheet_url: str, task_description: str) -> Dict[str, Any]:
            """
            Evaluates trainer performance based on GitLab repositories listed in a Google Sheet,
            against a given task description.
            """
            try:
                print(f"Starting evaluation. Sheet URL: {google_sheet_url}, Task: {task_description}")
                # 1. Fetch and parse data from Google Sheet
                trainee_infos = await self._fetch_google_sheet_data(google_sheet_url)
                if not trainee_infos:
                    return {"success": False, "error": "Could not retrieve or parse trainee data from Google Sheet."}
                
                print(f"Retrieved {len(trainee_infos)} trainees from sheet.")

                # 2. For each trainee, fetch GitLab repo content
                detailed_trainee_data = []
                for trainee in trainee_infos:
                    print(f"Fetching repo for {trainee['name']} from {trainee['repo_url']}")
                    try:
                        repo_content = await self._fetch_gitlab_repo_content(trainee['repo_url'])
                        detailed_trainee_data.append({
                            "name": trainee["name"],
                            "repo_url": trainee["repo_url"],
                            "repo_data": repo_content # This would be richer in a real implementation
                        })
                    except Exception as e:
                         print(f"Failed to fetch repo for {trainee['name']}: {e}")
                         detailed_trainee_data.append({
                            "name": trainee["name"],
                            "repo_url": trainee["repo_url"],
                            "repo_data": {"error": f"Could not fetch repo data: {e}"}
                        })
                
                if not detailed_trainee_data:
                     return {"success": False, "error": "No repository data could be fetched for any trainee."}

                # 3. Use AI to rank trainees based on repo data and task description
                prompt_sections = [f"Task Description:\n{task_description}\n"]
                prompt_sections.append("Trainee GitLab Repository Data:")
                for i, data in enumerate(detailed_trainee_data):
                    repo_summary = data['repo_data'].get('readme', 'No README or content fetched.')
                    if 'error' in data['repo_data']:
                        repo_summary = f"Error fetching repo: {data['repo_data']['error']}"
                    
                    prompt_sections.append(
                        f"\n--- Trainee {i+1} ---\n"
                        f"Name: {data['name']}\n"
                        f"Repo URL: {data['repo_url']}\n"
                        f"Repo Content Summary (e.g., README):\n{repo_summary[:1000]}...\n" # Truncate for prompt
                    )
                
                prompt_sections.append(
                    "\n--- Evaluation Request ---\n"
                    "Based on the provided task description and the summary of each trainee's GitLab repository (primarily their README or fetched content), "
                    "please provide a concise ranking of the trainees. \n"
                    "For each trainee, provide a brief justification for their rank. \n"
                    "The output should be a list of rankings, with the trainee's name, their rank, and a short justification.\n"
                    "Example: \n"
                    "1. Trainee Name X: Justification for rank 1...\n"
                    "2. Trainee Name Y: Justification for rank 2...\n"
                )
                
                full_prompt = "\n".join(prompt_sections)
                
                print(f"Sending prompt to AI for ranking (length: {len(full_prompt)} chars)")

                # Ensure this is an async call if the SDK supports it, or wrap in to_thread
                ai_response = await asyncio.to_thread(
                     self.ai_provider.generate_content, full_prompt
                )
                
                ranking_text = ai_response.text

                return {
                    "success": True,
                    "ranking_summary": ranking_text,
                    "trainees_processed": len(detailed_trainee_data),
                    "data_sources": {
                        "google_sheet_url": google_sheet_url,
                        "gitlab_repos": [td["repo_url"] for td in detailed_trainee_data]
                    }
                }

            except A2AIOException as e: # Custom exception for IO issues
                print(f"A2AIOException during evaluation: {e}")
                return {"success": False, "error": str(e)}
            except Exception as e:
                print(f"Unexpected exception during evaluation: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}

    def handle_task(self, task_input: Task) -> Task:
        """
        Handles an A2A task by invoking the 'evaluate_performance' MCP tool.
        It expects the task message content to be a JSON string with
        'google_sheet_url' and 'task_description'.
        """
        try:
            print(f"EvaluatorAgent: Received A2A task ID: {task_input.id}, Message type: {type(task_input.message)}")

            message_data = task_input.message # This is a dict as per the error

            if not message_data or not isinstance(message_data, dict):
                error_msg = "Task message is missing or not a dictionary as expected."
                print(error_msg)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg})
                task_input.artifacts = [ErrorContent(message=error_msg).to_dict()]
                return task_input

            content_data = message_data.get('content') # This should be a dict

            if not content_data or not isinstance(content_data, dict):
                error_msg = "Task message content is missing or not a dictionary."
                print(error_msg)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg})
                task_input.artifacts = [ErrorContent(message=error_msg).to_dict()]
                return task_input

            params = {}
            text_payload = None

            if content_data.get('type') == 'text':
                text_payload = content_data.get('text')
            
            if text_payload:
                try:
                    import json
                    payload = json.loads(text_payload)
                    if isinstance(payload, dict):
                        params["google_sheet_url"] = payload.get("google_sheet_url")
                        params["task_description"] = payload.get("task_description")
                    else:
                        raise ValueError("Parsed JSON from text payload is not a dictionary.")
                    
                    if not params.get("google_sheet_url") or not params.get("task_description"):
                        raise ValueError("Missing 'google_sheet_url' or 'task_description' in parsed JSON payload.")

                except (json.JSONDecodeError, ValueError) as e_parse:
                    error_msg_parse = f"Invalid text payload: Expected a JSON string with 'google_sheet_url' and 'task_description'. Error: {e_parse}"
                    print(error_msg_parse)
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_parse})
                    task_input.artifacts = [ErrorContent(message=error_msg_parse).to_dict()]
                    return task_input
            else:
                error_msg_content = "Task message content must be of type 'text' and contain a 'text' field."
                if not text_payload and content_data.get('type') == 'text': # Specifically if 'text' field was missing
                    error_msg_content = "Task message content is of type 'text' but is missing the 'text' field itself."
                print(error_msg_content)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_content})
                task_input.artifacts = [ErrorContent(message=error_msg_content).to_dict()]
                return task_input

            try:
                print(f"Calling MCP tool 'evaluate_performance' with params: {params}")
                mcp_result_obj = asyncio.run(self.mcp_server.call_tool("evaluate_performance", params=params))

                if not isinstance(mcp_result_obj, MCPResponse):
                    error_msg_unexpected = f"MCP tool call returned unexpected type: {type(mcp_result_obj)}"
                    print(error_msg_unexpected)
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_msg_unexpected})
                    task_input.artifacts = [ErrorContent(message=error_msg_unexpected).to_dict()]
                elif mcp_result_obj.is_error:
                    error_detail = "MCP tool execution indicated an error."
                    if mcp_result_obj.content and isinstance(mcp_result_obj.content, list) and len(mcp_result_obj.content) > 0:
                        first_content_item = mcp_result_obj.content[0]
                        if isinstance(first_content_item, dict) and 'text' in first_content_item:
                            error_detail = first_content_item['text']
                    print(f"MCP tool 'evaluate_performance' failed (is_error=True). Detail: {error_detail}")
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": str(error_detail)})
                    task_input.artifacts = [ErrorContent(message=str(error_detail)).to_dict()]
                else: # Not an error, so expect content to be the tool's result
                    try:
                        if not mcp_result_obj.content or not isinstance(mcp_result_obj.content, list) or len(mcp_result_obj.content) == 0:
                            raise ValueError("MCPResponse content is empty or not a list.")
                        
                        first_content_item = mcp_result_obj.content[0]
                        if not isinstance(first_content_item, dict) or first_content_item.get('type') != 'text' or 'text' not in first_content_item:
                            raise ValueError("MCPResponse content item is not a valid text dictionary or missing 'text' field.")

                        import json
                        actual_result_dict = json.loads(first_content_item['text'])
                        
                        if not isinstance(actual_result_dict, dict):
                            raise ValueError("Parsed content text from MCPResponse is not a dictionary.")

                        if actual_result_dict.get("success"):
                            print(f"MCP tool 'evaluate_performance' processed successfully. Result: {actual_result_dict}")
                            task_input.status = TaskStatus(state=TaskState.COMPLETED)
                            task_input.artifacts = [
                                FunctionResponseContent(
                                    name="performance_evaluation_result",
                                    response=actual_result_dict
                                ).to_dict()
                            ]
                        else: # success is False or missing in actual_result_dict
                            error_from_dict = actual_result_dict.get("error", "MCP tool reported failure without specific error message in result dictionary.")
                            print(f"MCP tool 'evaluate_performance' reported failure in its result. Detail: {error_from_dict}")
                            task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": str(error_from_dict)})
                            task_input.artifacts = [ErrorContent(message=str(error_from_dict)).to_dict()]

                    except (json.JSONDecodeError, ValueError, IndexError, KeyError) as e_content_parse:
                        error_detail_content = f"Failed to process/parse content from successful MCPResponse: {type(e_content_parse).__name__}: {e_content_parse}"
                        print(f"{error_detail_content}. Content: {mcp_result_obj.content if hasattr(mcp_result_obj, 'content') else 'No content attribute'}")
                        task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": error_detail_content})
                        task_input.artifacts = [ErrorContent(message=error_detail_content).to_dict()]

            except AttributeError as e_attr: # This might catch if mcp_result_obj doesn't have is_error or content initially
                import traceback
                error_trace_attr = traceback.format_exc()
                error_msg_attr = f"AttributeError during MCP tool call: {type(e_attr).__name__}: {e_attr}\\n{error_trace_attr}"
                print(error_msg_attr)
                task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": f"Internal error during MCP tool call: {type(e_attr).__name__}"})
                task_input.artifacts = [ErrorContent(message=error_msg_attr).to_dict()]
            
            return task_input

        except BaseException as be:
            import traceback
            critical_error_trace = traceback.format_exc()
            critical_error_msg = f"CRITICAL UNHANDLED EXCEPTION in handle_task: {type(be).__name__}: {be}\\n{critical_error_trace}"
            print(critical_error_msg)

            # Attempt to set status on task_input, assuming it's a valid Task object.
            try:
                if isinstance(task_input, Task):
                    task_input.status = TaskStatus(state=TaskState.FAILED, message={"error": f"Critical unhandled error: {type(be).__name__}"})
                    task_input.artifacts = [ErrorContent(message=critical_error_msg).to_dict()]
                else:
                    # This case should ideally not happen if task_input is always a Task.
                    # If it's not a Task, we can't set status/artifacts on it.
                    # The A2AServer will likely still have issues, but we've logged the root cause.
                    print("CRITICAL: task_input was not a Task object during BaseException handling.")
            except Exception as e_critical_set:
                print(f"CRITICAL: Failed to set error status on task_input during BaseException handling: {e_critical_set}")

            if isinstance(be, (KeyboardInterrupt, SystemExit)):
                raise be # Re-raise critical exit exceptions

            # Ensure a Task object is returned to prevent UnboundLocalError for 'result' in A2AServer
            # If task_input was not a Task or setting status failed, it might still be problematic upstream,
            # but returning it is better than letting the BaseException propagate raw if it's not critical for exit.
            if isinstance(task_input, Task):
                return task_input
        else:
                # If task_input is not a Task object, we are in a bad state.
                # Construct a dummy Task to satisfy A2AServer's need for a return value.
                # This is a last resort to prevent the UnboundLocalError in A2AServer.
                # The actual error is logged.
                try:
                    dummy_task_id = task_input.id if hasattr(task_input, 'id') else "unknown_task_due_to_critical_error"
                    return Task(id=dummy_task_id, status=TaskStatus(state=TaskState.FAILED, message={"error": f"Critical unhandled error: {type(be).__name__}"}))
                except Exception: # If even creating a dummy task fails
                    raise be # Re-raise the original BaseException as a last resort

    async def start_servers(self, a2a_port: int = 8004, mcp_port: int = 8005):
        print(f"Starting Evaluator Agent servers...")
        print(f"A2A Server: http://localhost:{a2a_port}")
        print(f"MCP Server (FastAPI): http://localhost:{mcp_port}")

        # A2A Server (using Flask, as per existing agents)
        from flask import Flask
        a2a_app = Flask(__name__)
        self.a2a_server = A2AServer(
            agent_card=self.agent_card, # Corrected: pass the method, not call it
            port=a2a_port
        )
        self.a2a_server.setup_routes(a2a_app)

        self.a2a_server.handle_task = self.handle_task
        # MCP Server (using FastAPI)
        mcp_fastapi_app = create_fastapi_app(self.mcp_server)
        
        # Run both servers concurrently
        # For Flask (a2a_app), run in a thread. For FastAPI (mcp_fastapi_app), uvicorn can run async.
        import uvicorn

        # Need to configure uvicorn to run the FastAPI app
        config = uvicorn.Config(mcp_fastapi_app, host="0.0.0.0", port=mcp_port, log_level="info")
        server = uvicorn.Server(config)
        
        await asyncio.gather(
            asyncio.to_thread(a2a_app.run, host='0.0.0.0', port=a2a_port, debug=False, use_reloader=False),
            server.serve() # Uvicorn's server.serve() is awaitable
        )

# Custom Exception for IO Operations for clarity
class A2AIOException(Exception):
    pass


if __name__ == "__main__":
    agent = EvaluatorAgent()
    try:
        asyncio.run(agent.start_servers())
    except KeyboardInterrupt:
        print("Evaluator Agent shutting down.")
