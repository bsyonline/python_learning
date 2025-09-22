from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    count: int
    id: str

def sub_node1(state: State):
    """sub_node1
    """
    print(f"sub_node1: {state['count']}")
    return {"count": state["count"] + 1}

def sub_node2(state: State):
    """sub_node2
    """
    print(f"sub_node2: {state['count']}")
    return {"count": state["count"] + 2}

def parent_node(state: State):
    """parent_node
    """
    print(f"parent_node: {state['count']}")
    return {"count": state["count"]+3}

sub_graph_builder = StateGraph(State)
sub_graph_builder.add_node("sub_node1", sub_node1)
sub_graph_builder.add_node("sub_node2", sub_node2)

sub_graph_builder.add_edge(START, "sub_node1")
sub_graph_builder.add_edge("sub_node1", "sub_node2")
sub_graph_builder.add_edge("sub_node2", END)

sub_graph = sub_graph_builder.compile()

parent_graph_builder = StateGraph(State)
parent_graph_builder.add_node("parent_node", parent_node)
parent_graph_builder.add_node("sub_graph", sub_graph)

parent_graph_builder.add_edge(START, "parent_node")
parent_graph_builder.add_edge("parent_node", "sub_graph")
parent_graph_builder.add_edge("sub_graph", END)

parent_graph = parent_graph_builder.compile()

print(parent_graph.invoke({"count": 1}))