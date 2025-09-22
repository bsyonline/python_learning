import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

async def main():
    print("🚀 Starting MCP Client Example...")
    
    # 初始化模型
    model = ChatOpenAI(
        model="deepseek-v3-1-250821",
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    # 设置MCP客户端，连接多个服务器
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["/Users/rolex/Dev/workspace/python/python_learning/third-party/langgraph/mcp/math_tool_mcp_server.py"],
                "transport": "stdio",
            },
            "host_info": {
                "command": "python",
                "args": ["/Users/rolex/Dev/workspace/python/python_learning/third-party/langgraph/mcp/host_info_tool_mcp_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    
    # 获取所有可用的工具
    tools = await client.get_tools()
    print(f"📋 Available tools: {[tool.name for tool in tools]}")
    
    # 将工具绑定到模型
    model_with_tools = model.bind_tools(tools)
    
    # 创建工具节点
    tool_node = ToolNode(tools)
    
    # 定义模型调用函数
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    # 构建图
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "call_model")
    
    # 编译图
    graph = builder.compile()
    
    # 测试用例
    test_cases = [
        "计算 (15 + 7) × 3 的结果",
        "北京现在的天气怎么样？",
        "计算半径为5的圆的面积",
        "比较北京和上海的天气",
        "计算10的阶乘",
        "我的电脑内存是多大"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 50)
        
        try:
            result = await graph.ainvoke({"messages": [{"role": "user", "content": query}]})
            
            # 提取最终回复
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                print(f"🤖 AI回复: {final_message.content}")
            else:
                print(f"📊 工具调用结果: {final_message}")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            print("💡 提示: 请确保MCP服务器正在运行:")
            print("  数学服务器: python math_tool_mcp_server.py")
            print("  天气服务器: python weather_tool_mcp_server.py")
            break
    
    print("\n✅ 示例完成!")

if __name__ == "__main__":
    asyncio.run(main())