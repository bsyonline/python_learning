from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

class ResponseFormatter(BaseModel):
    """始终使用此工具来构建对用户的响应。请用中文回答。"""
    answer: str = Field(description="用户问题的答案")
    followup_question: str = Field(description="用户可以提出的后续问题")
    confidence: float = Field(description="置信度，范围 0 到 1", ge=0, le=1)
    
class FactChecker(BaseModel):
    """使用此工具验证事实并提供结构化分析。请用中文回答。"""
    statement: str = Field(description="正在检查的陈述")
    is_true: bool = Field(description="陈述是否为真")
    explanation: str = Field(description="解释为什么是真或假")
    sources: list[str] = Field(description="用于验证的潜在来源")

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0
).bind_tools([ResponseFormatter, FactChecker])

ai_msg = llm.invoke("什么是RNN")
print(f"原始响应: {ai_msg}")
print("-" * 60)
try:
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"调用的工具: {tool_name}")
            print(f"参数: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")
            if tool_name == "ResponseFormatter":
                formatted_response = ResponseFormatter.model_validate(tool_args)
                print(f"解析后的响应: {formatted_response}")
            elif tool_name == "FactChecker":
                fact_check = FactChecker.model_validate(tool_args)
                print(f"事实检查: {fact_check}")
    else:
        print("未调用任何工具。原始响应:")
        print(ai_msg.content)
except Exception as e:
    print(f"错误: {e}")
