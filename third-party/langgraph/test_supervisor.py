from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

def add(a: float, b: float) -> float:
    '''Add two numbers.'''
    return a + b

def web_search(query: str) -> str:
    '''Search the web for information.'''
    return 'Here are the headcounts for each of the FAANG companies in 2024...'

model = ChatOpenAI(
    model="deepseek-v3-1-250821",
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0
)

math_agent = create_react_agent(
    model=model,
    tools=[add],
    name="math_expert",
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})
print(result)