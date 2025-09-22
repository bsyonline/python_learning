from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from typing import TypedDict

class State(TypedDict):
    count: int
    id: str

class ContextSchema(TypedDict):
    id: str

def node1(state: State, config: RunnableConfig):
    print(config["configurable"]["id"])
    return {"count": state["count"] + 1, "id": "node1"}

def node2(state: State, config: RunnableConfig):
    print(config["configurable"]["id"])
    return {"count": state["count"] + 1, "id": "node2"}

def condition(state: State):
    if state["count"] > 0:
        return "node2"
    else:
        return "node1"

def condition1(state: State):
    return state["count"] > 0

builder = StateGraph(state_schema=State, context_schema=ContextSchema)

builder.add_node("node1", node1)
builder.add_node("node2", node2)

builder.add_conditional_edges(START, condition)
builder.add_edge("node2", END)

graph = builder.compile()
print(graph.invoke({"count": 1}, {"configurable": {"id": "1"}}))


print("---")
builder = StateGraph(State, context_schema=ContextSchema)

builder.add_node("node1", node1)
builder.add_node("node2", node2)

builder.add_conditional_edges(START, condition1, path_map={True: "node1", False: "node2"})
builder.add_edge("node2", END)

graph = builder.compile()
print(graph.invoke({"count": 2}, {"configurable": {"id": "1"}}))
