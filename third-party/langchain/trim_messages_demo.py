"""
LangChain 消息修剪演示
基于官方文档: https://python.langchain.com/docs/how_to/trim_messages/

这个演示展示了如何使用LangChain的trim_messages功能来管理聊天历史长度，
确保消息不会超过模型的上下文窗口限制。
"""

from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage, 
    ToolMessage, 
    trim_messages
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

# 初始化LLM
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0
)

def demo_basic_trimming():
    """演示基础消息修剪功能"""
    print("\n🚀 演示 1: 基础消息修剪")
    print("=" * 50)
    
    # 创建一个长的聊天历史
    messages = [
        SystemMessage("你是一个有用的AI助手，请用中文回答用户的问题。"),
        HumanMessage("什么是机器学习？"),
        AIMessage("机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需显式编程。"),
        HumanMessage("机器学习有哪些主要类型？"),
        AIMessage("机器学习主要分为三类：监督学习、无监督学习和强化学习。"),
        HumanMessage("能详细解释一下监督学习吗？"),
        AIMessage("监督学习使用标记数据训练模型，模型学习输入特征与输出标签之间的映射关系。"),
        HumanMessage("那无监督学习呢？"),
        AIMessage("无监督学习使用未标记数据，模型自主发现数据中的模式和结构，如聚类和降维。"),
        HumanMessage("强化学习又是什么？"),
        AIMessage("强化学习通过试错学习，智能体在环境中采取行动并获得奖励，目标是最大化累积奖励。"),
        HumanMessage("这些学习方法各有什么应用场景？"),
        AIMessage("监督学习用于分类和回归，无监督学习用于聚类和异常检测，强化学习用于游戏和机器人控制。"),
        HumanMessage("现在请总结一下机器学习的核心概念")
    ]
    
    print("📋 原始消息历史 ({}条消息):".format(len(messages)))
    for i, msg in enumerate(messages):
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {msg.content[:50]}...")
    
    # 计算原始消息的token数量
    original_tokens = count_tokens_approximately(messages)
    print(f"\n🔢 原始消息token数量: {original_tokens}")
    
    # 修剪消息，保留最近的45个token
    trimmed_messages = trim_messages(
        messages,
        # 保留最后 <= max_tokens 个token的消息
        strategy="last",
        # 使用token计数器
        token_counter=count_tokens_approximately,
        # 设置最大token数量
        max_tokens=45,
        # 聊天历史应该以HumanMessage开始
        start_on="human",
        # 聊天历史应该以HumanMessage或ToolMessage结束
        end_on=("human", "tool"),
        # 包含系统消息（如果存在）
        include_system=True,
        allow_partial=False,
    )
    
    print("\n✂️  修剪后的消息历史 ({}条消息):".format(len(trimmed_messages)))
    for i, msg in enumerate(trimmed_messages):
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {msg.content[:50]}...")
    
    # 计算修剪后消息的token数量
    trimmed_tokens = count_tokens_approximately(trimmed_messages)
    print(f"\n🔢 修剪后消息token数量: {trimmed_tokens}")
    print(f"📉 减少了 {original_tokens - trimmed_tokens} 个token")

def demo_message_count_trimming():
    """演示基于消息数量的修剪"""
    print("\n\n🚀 演示 2: 基于消息数量的修剪")
    print("=" * 50)
    
    # 创建一个长的聊天历史
    messages = [
        SystemMessage("你是一个幽默的AI助手，请用中文回答并加入一些幽默元素。"),
        HumanMessage("为什么程序员总是分不清万圣节和圣诞节？"),
        AIMessage("因为Oct 31 == Dec 25！哈哈哈！"),
        HumanMessage("那为什么程序员不喜欢大自然？"),
        AIMessage("因为他们习惯了bug，但不喜欢真正的虫子！"),
        HumanMessage("程序员最喜欢的健身方式是什么？"),
        AIMessage("当然是举重啦！不过举的是笔记本电脑的重量！"),
        HumanMessage("程序员怎么喝咖啡？"),
        AIMessage("他们先写一个喝咖啡的函数，然后调试它直到能正常运行！"),
        HumanMessage("程序员为什么喜欢黑暗？"),
        AIMessage("因为光会吸引bug！哦，我说的是真正的虫子和代码bug都喜欢光！"),
        HumanMessage("现在给我讲一个关于AI的笑话")
    ]
    
    print("📋 原始消息历史 ({}条消息):".format(len(messages)))
    for i, msg in enumerate(messages):
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {msg.content[:40]}...")
    
    # 基于消息数量修剪（每条消息算作1个"token"）
    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        # 使用len作为token计数器，每条消息计为1
        token_counter=len,
        # 最多保留5条消息
        max_tokens=5,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )
    
    print("\n✂️  修剪后的消息历史 ({}条消息):".format(len(trimmed_messages)))
    for i, msg in enumerate(trimmed_messages):
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {msg.content[:40]}...")

def demo_with_tool_messages():
    """演示包含工具消息的修剪"""
    print("\n\n🚀 演示 3: 包含工具消息的修剪")
    print("=" * 50)
    
    # 创建一个包含工具调用的聊天历史
    messages = [
        SystemMessage("你是一个有帮助的AI助手，可以使用工具来回答问题。"),
        HumanMessage("现在几点了？"),
        AIMessage("", tool_calls=[{"id": "call_123", "name": "get_current_time", "args": {}}]),
        ToolMessage("2024-01-15 14:30:25", tool_call_id="call_123"),
        HumanMessage("那计算一下3 + 5等于多少"),
        AIMessage("", tool_calls=[{"id": "call_456", "name": "add", "args": {"a": 3, "b": 5}}]),
        ToolMessage("8", tool_call_id="call_456"),
        HumanMessage("再计算一下10 * 2"),
        AIMessage("", tool_calls=[{"id": "call_789", "name": "multiply", "args": {"a": 10, "b": 2}}]),
        ToolMessage("20", tool_call_id="call_789"),
        HumanMessage("现在请告诉我今天的日期和这些计算结果")
    ]
    
    print("📋 原始消息历史 (包含工具消息):")
    for i, msg in enumerate(messages):
        content_preview = msg.content[:30] if msg.content else str(msg.tool_calls)[:30] if hasattr(msg, 'tool_calls') else "[工具消息]"
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {content_preview}...")
    
    # 修剪消息，确保工具消息的正确性
    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=60,
        start_on="human",
        # 允许以HumanMessage或ToolMessage结束
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )
    
    print("\n✂️  修剪后的消息历史:")
    for i, msg in enumerate(trimmed_messages):
        content_preview = msg.content[:30] if msg.content else str(msg.tool_calls)[:30] if hasattr(msg, 'tool_calls') else "[工具消息]"
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {content_preview}...")

def demo_model_specific_trimming():
    """演示针对特定模型的修剪"""
    print("\n\n🚀 演示 4: 针对特定模型的修剪")
    print("=" * 50)
    
    # 创建一个长的聊天历史
    messages = [
        SystemMessage("你是一个专业的AI助手，请用中文提供详细的技术解释。"),
        HumanMessage("请解释一下Transformer模型的工作原理"),
        AIMessage("Transformer模型基于自注意力机制，它通过计算输入序列中每个位置与其他所有位置的相关性来捕捉长距离依赖关系。"),
        HumanMessage("自注意力机制具体是怎么工作的？"),
        AIMessage("自注意力机制通过查询(Query)、键(Key)和值(Value)三个矩阵来计算注意力权重，然后对值进行加权求和。"),
        HumanMessage("那多头注意力又是什么？"),
        AIMessage("多头注意力将输入投影到多个子空间，每个头学习不同的表示，最后将结果拼接起来，增强了模型的表示能力。"),
        HumanMessage("Transformer还有哪些重要组件？"),
        AIMessage("还包括位置编码、前馈神经网络、层归一化和残差连接等组件，它们共同构成了Transformer的架构。"),
        HumanMessage("这些组件各有什么作用？"),
        AIMessage("位置编码提供序列顺序信息，前馈网络进行非线性变换，层归一化稳定训练，残差连接缓解梯度消失。"),
        HumanMessage("现在请总结一下Transformer的核心优势")
    ]
    
    print("📋 原始消息历史:")
    for i, msg in enumerate(messages):
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {msg.content[:40]}...")
    
    # 使用通用token计数器进行修剪
    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        # 使用通用token计数器
        token_counter=count_tokens_approximately,
        # 设置合理的最大token数量
        max_tokens=100,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )
    
    print("\n✂️  修剪后的消息:")
    for i, msg in enumerate(trimmed_messages):
        print(f"  {i+1:2d}. {type(msg).__name__:15s}: {msg.content[:40]}...")
    
    # 计算修剪后的token数量
    trimmed_tokens = count_tokens_approximately(trimmed_messages)
    print(f"\n🔢 修剪后消息token数量: {trimmed_tokens}")

def main():
    """主函数"""
    print("🔧 LangChain 消息修剪演示")
    print("=" * 60)
    print("基于官方文档: https://python.langchain.com/docs/how_to/trim_messages/")
    print("=" * 60)
    print("📚 演示内容:")
    print("  • 基础消息修剪 (基于token数量)")
    print("  • 基于消息数量的修剪")
    print("  • 包含工具消息的修剪")
    print("  • 针对特定模型的修剪")
    print("=" * 60)
    
    # 运行所有演示
    demo_basic_trimming()
    demo_message_count_trimming()
    demo_with_tool_messages()
    demo_model_specific_trimming()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成!")
    print("💡 关键知识点:")
    print("  • trim_messages用于管理聊天历史长度")
    print("  • 支持基于token数量或消息数量的修剪")
    print("  • 确保修剪后的消息历史格式正确")
    print("  • 可以处理包含工具消息的复杂场景")
    print("  • 支持针对特定模型的token计数")
    print("\n🚀 使用方法:")
    print("cd /Users/rolex/Dev/workspace/python/python_learning/third-party/langchain")
    print("python trim_messages_demo.py")

if __name__ == "__main__":
    main()