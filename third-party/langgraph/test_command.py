from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    count: int
    id: str

def node1(state: State):
    """node1
    """
    print(f"node1: {state['count']}")
    if state["count"] > 0:
        return Command(goto="node2", update={"count": state["count"] + 1})
    return Command(goto="node3", update={"count": state["count"] + 1})

def node2(state: State):
    """node2
    """
    print(f"node2: {state['count']}")
    return Command(goto=END, update={"count": state["count"] + 2})

def node3(state: State):
    """node3
    """
    print(f"node3: {state['count']}")
    return Command(goto=END, update={"count": state["count"] + 3})

builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)

builder.add_edge(START, "node1")

graph = builder.compile()

print(graph.invoke(Command(update={"count": 2}), {"configurable": {"id": "1"}}))
