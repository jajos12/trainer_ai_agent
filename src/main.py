import asyncio
import os
from dotenv import load_dotenv
from agents.coordinator import CoordinatorAgent
from agents.curator import CuratorAgent
from mcp_servers.curator_mcp.server import CuratorMCPServer

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get API keys and configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize agents and servers
    curator_mcp = CuratorMCPServer()
    curator_agent = CuratorAgent()
    coordinator = CoordinatorAgent(openai_api_key=openai_api_key)
    
    try:
        # Start MCP server
        await curator_mcp.start()
        print("Curator MCP server started")
        
        # Start Curator agent
        await curator_agent.start()
        print("Curator agent started")
        
        # Discover agents
        await coordinator.discover_agents()
        print("Agents discovered")
        
        # Example interaction
        response = await coordinator.handle_user_request(
            "Find me the latest trends in AI and create a learning task"
        )
        print(f"Response: {response}")
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        await curator_mcp.stop()
        await curator_agent.stop()

if __name__ == "__main__":
    asyncio.run(main()) 