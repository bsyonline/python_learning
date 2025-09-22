from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated, List, TypedDict
from operator import add

class State(TypedDict):
    count: Annotated[int, add]
    arr: List[int]
    id: str

class ContextSchema(TypedDict):
    id: str

def node1(state: State)-> State:
    """node1
    """
    print(f"node1: {state['count']}\n")
    return {"count": state["count"]}

def node2(state: State)-> State:
    """node2
    """
    print(f"node2: {state['count']}\n")
    return {"count": state["count"]}

def condition(state: State):
    """condition
    """
    result = []
    for i in state["arr"]:
        if i % 2 == 0:
            result.append(Send("node1", {"count": i, "id": "node1"}))
        else:
            result.append(Send("node2", {"count": i, "id": "node2"}))
    return result

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("node1", node1)
builder.add_node("node2", node2)

builder.add_conditional_edges(START, condition, ["node1", "node2"])
builder.add_edge("node1", END)
builder.add_edge("node2", END)

graph = builder.compile()
print(graph.invoke({"arr": [1,2,3,4,5,6], "count": 0}))
