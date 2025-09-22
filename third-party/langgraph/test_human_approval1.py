from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, List, Literal, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage

from operator import add
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

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add]

def call_llm(state: State):
    """call_llm
    """
    print(state["messages"])
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def human_approval(state: State) -> Command[Literal["call_llm", END]]:
    """human_approval
    """
    is_approval = interrupt({"question": "是否同意调用llm？(y/n)"})
    print(f"is_approval: {is_approval}")
    if is_approval == True:
        return Command(goto="call_llm")
    return Command(goto=END)

builder = StateGraph(State)
builder.add_node("call_llm", call_llm)
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "human_approval")

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

print(graph.invoke(
    input={"messages": [HumanMessage("今天星期几")]}, 
    config=config
))
user_input = input("是否继续执行？(y/n): ").lower()
if user_input == "y":
    print(graph.invoke(Command(resume=True), config=config))
else:
    print(graph.invoke(Command(resume=False), config=config))