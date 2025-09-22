from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_verbose
import asyncio
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

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手，请用中文回答用户的问题。"),
    ("user", "{question}")
])

output_parser = StrOutputParser()

chain = prompt_template | llm | output_parser

async def async_stream():
    chunks = []
        
    # 使用同步 stream 方法
    async for chunk in chain.astream({"question": "什么是CNN"}):
        chunks.append(chunk)
        print(chunk, end="", flush=True)  # 实时显示每个chunk
        await asyncio.sleep(0.01)  # 稍微延迟以便观察流式效果

asyncio.run(async_stream())
