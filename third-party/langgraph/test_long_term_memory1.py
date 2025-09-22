from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url
)

# 定义状态结构
class AgentState(MessagesState):
    # 我们使用 MessagesState 中的 messages 字段来存储对话历史
    # 这将作为我们的长期记忆存储
    pass

# 定义节点函数
def chatbot_node(state: AgentState):
    """聊天机器人节点，使用模型生成回复"""
    response = model.invoke(state["messages"])
    return {"messages": response}

# 创建状态图
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("chatbot", chatbot_node)

# 添加边
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 配置，使用固定的线程ID来保持对话状态
config = {"configurable": {"thread_id": "1"}}

# 测试函数
def run_long_term_memory_example():
    # 使用文件数据库作为检查点存储（真正的长期记忆）
    with SqliteSaver.from_conn_string("memory.db") as checkpointer:
        # 编译图
        graph = builder.compile(checkpointer=checkpointer)

        input_message = {"role": "user", "content": "中国的首都是哪里"}
        print(f"用户: {input_message['content']}")
        
        # 调用图
        result = graph.invoke({"messages": [input_message]}, config)
        print(f"助手: {result['messages'][-1].content}")

# 运行示例
if __name__ == "__main__":
    run_long_term_memory_example()