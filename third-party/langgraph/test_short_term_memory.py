from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url
)

def llm_node(state: MessagesState):
    """llm_node
    """
    response = llm.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)

builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

# print(graph.invoke({"messages": [{"role": "user", "content": "你是谁？"}]}))

for event in graph.stream(
    input={"messages": [{"role": "user", "content": "中国的首都是哪里"}]}, 
    config=config, 
    stream_mode="values"
):
    print(event["messages"])

for event in graph.stream(
    input={"messages": [{"role": "user", "content": "法国呢"}]}, 
    config=config, 
    stream_mode="values"
):
    print(event["messages"])