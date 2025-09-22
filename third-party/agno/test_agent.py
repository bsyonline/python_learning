from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.tools.mcp import MCPTools
from agno.models.openai.like import OpenAILike
from os import getenv
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    name="Agno Agent",
    model=OpenAILike(
        id="deepseek-v3-1-250821",
        api_key=getenv("API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    db=SqliteDb(db_file="agno.db"),
    # Add the Agno MCP server to the Agent
    tools=[MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")],
    # Add the previous session history to the context
    add_history_to_context=True,
    markdown=True,
)

# Create the AgentOS
# agent_os = AgentOS(agents=[agent])
# Get the FastAPI app for the AgentOS
# app = agent_os.get_app()

print(agent.run("介绍一下Agno"))