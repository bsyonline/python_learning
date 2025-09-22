import asyncio
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 添加其他状态字段以便更好地演示values模式
    user_name: str
    conversation_step: int
    is_processing: bool


def create_demo_graph():
    """创建演示图"""
    
    def greeting_node(state: State):
        """问候节点"""
        user_name = state.get("user_name", "用户")
        return {
            "messages": [AIMessage(content=f"你好，{user_name}！欢迎使用我们的聊天机器人。")],
            "conversation_step": state.get("conversation_step", 0) + 1,
            "is_processing": False
        }
    
    def help_node(state: State):
        """帮助节点"""
        return {
            "messages": [AIMessage(content="我可以帮助你了解LangGraph的流式传输模式。")],
            "conversation_step": state.get("conversation_step", 0) + 1,
            "is_processing": False
        }
    
    # 创建图构建器
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("greeting", greeting_node)
    builder.add_node("help", help_node)
    
    # 添加条件边
    def route_message(state: State):
        if "你好" in state["messages"][-1].content:
            return "greeting"
        else:
            return "help"
    
    builder.add_conditional_edges(
        START,
        route_message,
        {
            "greeting": "greeting",
            "help": "help"
        }
    )
    
    # 添加结束边
    builder.add_edge("greeting", END)
    builder.add_edge("help", END)
    
    # 编译图
    graph = builder.compile()
    
    return graph


async def demonstrate_values_mode():
    """演示values模式"""
    print("=== Values模式演示 ===")
    print("特点：传输整个状态的更新\n")
    
    # 创建图
    graph = create_demo_graph()
    
    # 准备输入状态
    inputs = {
        "messages": [HumanMessage(content="你好！")],
        "user_name": "张三",
        "conversation_step": 0,
        "is_processing": True
    }
    
    print("初始状态:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")
    print()
    
    print("Values模式流式传输:")
    async for chunk in graph.astream(inputs, stream_mode="values"):
        print(f"收到状态更新: {type(chunk)}")
        print("  包含的字段:", list(chunk.keys()))
        if "messages" in chunk and chunk["messages"]:
            print(f"  最后一条消息: {chunk['messages'][-1].content}")
        if "user_name" in chunk:
            print(f"  用户名: {chunk['user_name']}")
        if "conversation_step" in chunk:
            print(f"  对话步骤: {chunk['conversation_step']}")
        if "is_processing" in chunk:
            print(f"  处理状态: {chunk['is_processing']}")
        print()


async def demonstrate_messages_mode():
    """演示messages模式"""
    print("=== Messages模式演示 ===")
    print("特点：专门传输消息\n")
    
    # 创建图
    graph = create_demo_graph()
    
    # 准备输入状态
    inputs = {
        "messages": [HumanMessage(content="你好！")],
        "user_name": "张三",
        "conversation_step": 0,
        "is_processing": True
    }
    
    print("初始状态:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")
    print()
    
    print("Messages模式流式传输:")
    async for chunk in graph.astream(inputs, stream_mode="messages"):
        print(f"收到消息更新: {type(chunk)}")
        print(f"  消息数量: {len(chunk)}")
        for i, message in enumerate(chunk):
            # 处理不同类型的消息对象
            if hasattr(message, 'content'):
                print(f"  消息 {i+1}: [{type(message).__name__}] {message.content}")
            else:
                print(f"  消息 {i+1}: {message}")
        print()


async def demonstrate_updates_mode():
    """演示updates模式"""
    print("=== Updates模式演示 ===")
    print("特点：传输状态的增量更新\n")
    
    # 创建图
    graph = create_demo_graph()
    
    # 准备输入状态
    inputs = {
        "messages": [HumanMessage(content="你好！")],
        "user_name": "张三",
        "conversation_step": 0,
        "is_processing": True
    }
    
    print("初始状态:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")
    print()
    
    print("Updates模式流式传输:")
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        print(f"收到更新: {type(chunk)}")
        print(f"  更新来自节点: {list(chunk.keys())[0] if chunk else '未知'}")
        node_name = list(chunk.keys())[0] if chunk else None
        if node_name:
            node_update = chunk[node_name]
            print(f"  节点 '{node_name}' 的更新:")
            for key, value in node_update.items():
                print(f"    {key}: {value}")
        print()


async def main():
    """主函数"""
    print("LangGraph Streaming Modes Comparison\n")
    
    # 演示values模式
    await demonstrate_values_mode()
    
    print("-" * 50)
    
    # 演示updates模式
    await demonstrate_updates_mode()
    
    print("-" * 50)
    
    # 演示messages模式
    await demonstrate_messages_mode()
    
    # 总结区别
    print("=" * 50)
    print("关键区别总结:")
    print("=" * 50)
    print("""
1. 数据内容:
   - Values模式: 传输完整的状态对象，包含所有字段
   - Updates模式: 传输状态的增量更新，只包含变化的字段
   - Messages模式: 只传输消息列表，不包含其他状态字段

2. 使用场景:
   - Values模式: 适用于需要访问完整状态信息的复杂工作流
   - Updates模式: 适用于需要监控状态变化的场景
   - Messages模式: 专为聊天应用优化，只关注消息流

3. 数据量:
   - Values模式: 数据量较大，包含所有状态信息
   - Updates模式: 数据量适中，只包含变化的状态
   - Messages模式: 数据量较小，只包含消息

4. 处理方式:
   - Values模式: 需要处理完整的状态字典
   - Updates模式: 处理增量更新字典
   - Messages模式: 直接处理消息列表

选择建议:
- 如果你只需要显示聊天消息 → 使用 Messages 模式
- 如果你需要监控状态变化 → 使用 Updates 模式
- 如果你需要监控整个工作流状态 → 使用 Values 模式
- 如果你在构建聊天机器人 → Messages 模式更适合
- 如果你在构建复杂的状态机 → Values 模式更合适
    """)


# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())