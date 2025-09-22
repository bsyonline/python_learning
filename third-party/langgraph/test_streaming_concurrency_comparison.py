import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState

class State(MessagesState):
    pass

def create_graph():
    """创建一个简单的图用于演示"""
    def slow_chatbot(state: State):
        # 模拟需要一些处理时间的聊天机器人
        user_message = state["messages"][-1].content
        # 模拟处理延迟
        time.sleep(0.1)
        response = f"回复: '{user_message}'"
        return {"messages": [AIMessage(content=response)]}
    
    # 定义图
    builder = StateGraph(State)
    builder.add_node("slow_chatbot", slow_chatbot)
    builder.set_entry_point("slow_chatbot")
    graph = builder.compile()
    
    return graph


# 同步顺序处理
def sync_sequential_processing():
    """同步顺序处理多个请求"""
    print("=== 同步顺序处理 ===")
    graph = create_graph()
    
    # 创建多个请求
    messages = [
        HumanMessage(content=f"消息 {i}") 
        for i in ['你好', '你是谁', '你从哪来', '要到哪去', '再见']
    ]
    
    start_time = time.time()
    
    # 逐个处理请求（阻塞式）
    responses = []
    for i, message in enumerate(messages):
        print(f"开始处理请求 {i+1}...")
        inputs = {"messages": [message]}
        
        # 同步处理
        response = None
        for chunk in graph.stream(inputs, stream_mode="messages"):
            for msg in chunk:
                if isinstance(msg, AIMessage):
                    response = msg.content
        responses.append((i+1, response))
        print(f"请求 {i+1} 处理完成")
    
    end_time = time.time()
    
    # 显示结果
    for i, response in responses:
        print(f"请求 {i} 响应: {response}")
    
    total_time = end_time - start_time
    print(f"同步顺序处理总耗时: {total_time:.2f}秒\n")
    return total_time


# 同步并发处理（使用线程池）
def sync_concurrent_processing():
    """同步并发处理多个请求（使用线程池）"""
    print("=== 同步并发处理（线程池） ===")
    graph = create_graph()
    
    # 创建多个请求
    messages = [
        HumanMessage(content=f"消息 {i}") 
        for i in ['你好', '你是谁', '你从哪来', '要到哪去', '再见']
    ]
    
    def process_request(i, message):
        """处理单个请求"""
        print(f"开始处理请求 {i}...")
        inputs = {"messages": [message]}
        
        # 同步处理
        response = None
        for chunk in graph.stream(inputs, stream_mode="messages"):
            for msg in chunk:
                if isinstance(msg, AIMessage):
                    response = msg.content
        
        print(f"请求 {i} 处理完成")
        return i, response
    
    start_time = time.time()
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交所有任务
        futures = [
            executor.submit(process_request, i+1, message) 
            for i, message in enumerate(messages)
        ]
        
        # 获取结果
        results = [future.result() for future in futures]
    
    end_time = time.time()
    
    # 显示结果
    for i, response in results:
        print(f"请求 {i} 响应: {response}")
    
    total_time = end_time - start_time
    print(f"同步并发处理总耗时: {total_time:.2f}秒\n")
    return total_time


# 异步并发处理
async def async_concurrent_processing():
    """异步并发处理多个请求"""
    print("=== 异步并发处理 ===")
    graph = create_graph()
    
    # 创建多个请求
    messages = [
        HumanMessage(content=f"消息 {i}") 
        for i in ['你好', '你是谁', '你从哪来', '要到哪去', '再见']
    ]
    
    async def process_request(i, message):
        """异步处理单个请求"""
        print(f"开始处理请求 {i}...")
        inputs = {"messages": [message]}
        
        # 异步处理
        response = None
        async for chunk in graph.astream(inputs, stream_mode="messages"):
            for msg in chunk:
                if isinstance(msg, AIMessage):
                    response = msg.content
        
        print(f"请求 {i} 处理完成")
        return i, response
    
    start_time = time.time()
    
    # 并发处理所有请求
    tasks = [
        process_request(i+1, message) 
        for i, message in enumerate(messages)
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    # 显示结果
    for i, response in results:
        print(f"请求 {i} 响应: {response}")
    
    total_time = end_time - start_time
    print(f"异步并发处理总耗时: {total_time:.2f}秒\n")
    return total_time


def main():
    """主函数 - 运行所有对比示例"""
    print("LangGraph 同步与异步并发处理对比示例\n")
    
    # 运行同步顺序处理
    sync_seq_time = sync_sequential_processing()
    
    # 运行同步并发处理
    sync_con_time = sync_concurrent_processing()
    
    # 运行异步并发处理
    async_con_time = asyncio.run(async_concurrent_processing())
    
    # 性能对比总结
    print("=== 性能对比总结 ===")
    print(f"同步顺序处理耗时: {sync_seq_time:.2f}秒")
    print(f"同步并发处理耗时: {sync_con_time:.2f}秒")
    print(f"异步并发处理耗时: {async_con_time:.2f}秒")
    print()
    
    # 计算性能提升
    if sync_con_time > 0:
        sync_improvement = sync_seq_time / sync_con_time
        print(f"同步并发相比同步顺序处理性能提升: {sync_improvement:.1f}倍")
    
    if async_con_time > 0:
        async_improvement = sync_seq_time / async_con_time
        print(f"异步并发相比同步顺序处理性能提升: {async_improvement:.1f}倍")
    
    if sync_con_time > 0 and async_con_time > 0:
        comparison = sync_con_time / async_con_time
        print(f"异步并发相比同步并发处理性能提升: {comparison:.1f}倍")
    
    print()
    print("=== 结论 ===")
    print("1. 同步顺序处理：逐个处理请求，总耗时等于所有请求处理时间之和")
    print("2. 同步并发处理：使用线程池并发处理请求，显著减少总耗时")
    print("3. 异步并发处理：使用异步方法并发处理请求，性能最佳")
    print("4. 在需要处理多个请求的场景中，推荐使用异步并发处理")


if __name__ == "__main__":
    main()