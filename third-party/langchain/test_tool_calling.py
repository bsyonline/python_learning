from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    print(f"tool call: add {a} and {b}")
    return a + b

llm=ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
).bind_tools([add])

result = llm.invoke("1 + 2等于几")
print("原始响应:")
print(result)
print("\n" + "="*50)

# 检查是否有工具调用
if hasattr(result, 'tool_calls') and result.tool_calls:
    print("检测到工具调用!")
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"调用的工具: {tool_name}")
        print(f"参数: {tool_args}")
        
        # 执行工具调用
        if tool_name == "add":
            a = tool_args["a"]
            b = tool_args["b"]
            result_value = add(a, b)
            print(f"工具执行结果: {a} + {b} = {result_value}")
            prompt = f"""
                用户问题: 1 + 2等于几
                工具执行结果: {result_value}
                请基于工具调用结果给用户一个完整的回答。
            """
            response = llm.invoke(prompt)
            print(f"最终响应: {response.content}")
else:
    print("没有检测到工具调用")
    print("直接回答内容:")
    print(result.content)

