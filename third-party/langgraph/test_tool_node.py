from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")



@tool
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    print(f"weather tool")
    return f"It's always sunny in {city}!"

@tool
def calculator(expression: str):
    """Evaluate a mathematical expression."""
    print(f"calculator tool")
    try:
        # 安全警告：在生产环境中切勿使用eval，这里仅为演示
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except:
        return "Error: Invalid expression."

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url
).bind_tools([get_weather, calculator])

def llm_node(state: AgentState) -> AgentState:
    """LLM node"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def select_tools(state: AgentState) -> str:
    """根据当前状态选择要调用的工具"""
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "llm"
    elif isinstance(last_message, ToolMessage):
        return "tools"
    else:
        return "end"

builder = StateGraph(AgentState)

tool_node = ToolNode(tools=[get_weather, calculator])
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", select_tools, {
    "llm": "llm",
    "tools": "tools",
    "end": END
})
builder.add_edge("tools", "llm")

graph = builder.compile()

inputs = {"messages": [HumanMessage("What is 123 * 45? And what is the weather in sf?")]}
outputs = graph.invoke(inputs)
print(f"{outputs}\n")
for i, message in enumerate(outputs["messages"]):
    print(f"{i+1}. [{message.type.upper()}]: {message.content}")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"   -> Tool Call: {tool_call['name']}({tool_call['args']})")