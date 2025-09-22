from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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

system_template = "请用中文回答，并以JSON格式输出你的回答"

prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。请始终使用符合提供模式的有效 JSON 进行响应。请用中文回答。"),
        ("user", "{question}")
    ])

chain = prompt_template | llm | JsonOutputParser()
print(chain.invoke({"question": "什么是langchain"}))
