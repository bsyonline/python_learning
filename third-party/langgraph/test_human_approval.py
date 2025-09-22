from typing import TypedDict
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from langgraph.types import interrupt, Command


class State(TypedDict):
    some_text: str


def human_node(state: State):
    value = interrupt(
        {
            "text_to_revise": state["some_text"]
        }
    )
    return {
        "some_text": value
    }


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")
graph_builder.add_edge("human_node", END)
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["human_node"])
# Pass a thread ID to the graph to run it.
config = {"configurable": {"thread_id": uuid.uuid4()}}
# Run the graph until the interrupt is hit.
result = graph.invoke({"some_text": "original text"}, config=config)

is_approval = input({"question": "是否同意？(y/n)"})
if is_approval.lower() == "y":
    result = graph.invoke(Command(resume=True), config=config)
    print(result)