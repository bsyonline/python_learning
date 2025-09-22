from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

class State(TypedDict):
    count: int
    id: str

class ContextSchema(TypedDict):
    id: str

def node1(state: State, config: RunnableConfig)-> State:
    """node1
    """
    print(config["configurable"]["id"])
    return {"count": state["count"] + 1, "id": config["configurable"]["id"]}

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("node1", node1)

builder.add_edge(START, "node1")
builder.add_edge("node1", END)

graph = builder.compile()

print(graph.invoke({"count": 0}, {"configurable": {"id": "1"}}))
print(graph.invoke({"count": 0}, {"configurable": {"id": "2"}}))

print("---")
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("node1", node1, cache_policy=CachePolicy(ttl=10))

builder.add_edge(START, "node1")
builder.add_edge("node1", END)

graph = builder.compile(cache=InMemoryCache())

print(graph.invoke({"count": 0}, {"configurable": {"id": "1"}}))
print(graph.invoke({"count": 0}, {"configurable": {"id": "2"}}))