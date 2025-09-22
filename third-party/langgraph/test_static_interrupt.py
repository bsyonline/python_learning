from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import time


class State(TypedDict):
    input: str
    output: str
    step: int


def step_1(state: State) -> State:
    """第一步：接收输入并进行初步处理"""
    print("--- 执行步骤 1 ---")
    print(f"接收到输入: {state['input']}")
    # 模拟一些处理时间
    time.sleep(1)
    return {
        "output": f"步骤1处理了: {state['input']}",
        "step": 1
    }


def step_2(state: State) -> State:
    """第二步：进一步处理"""
    print("--- 执行步骤 2 ---")
    print(f"当前状态: {state['output']}")
    # 模拟一些处理时间
    time.sleep(1)
    return {
        "output": f"步骤2处理了: {state['output']}",
        "step": 2
    }


def step_3(state: State) -> State:
    """第三步：最终处理"""
    print("--- 执行步骤 3 ---")
    print(f"当前状态: {state['output']}")
    # 模拟一些处理时间
    time.sleep(1)
    return {
        "output": f"最终结果: {state['output']}",
        "step": 3
    }


# 创建图
builder = StateGraph(State)

# 添加节点
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

# 添加边
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# 设置内存检查点
memory = InMemorySaver()

# 编译图并设置静态中断点
# 在执行step_2之前中断
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["step_2"]  # 静态中断：在执行step_2之前暂停
)

config = {"configurable": {"thread_id": "static-interrupt-example"}}

result = graph.invoke(
    {"input": "Hello LangGraph", "output": "", "step": 0}, 
    config
)

print(f"第一次执行结果: {result}")
print(f"当前步骤: {result['step']}")

snapshot = graph.get_state(config)
print(f"下一个要执行的节点: {snapshot.next}")

if snapshot.next:
    print("在step_2之前暂停,需要人工确认才能继续...")
    
    # 等待用户输入
    user_input = input("是否继续执行下一步？(y/n): ")
    
    if user_input.lower() == "y":
        print("\n=== 恢复执行 ===")
        # 恢复执行
        result = graph.invoke(Command(resume=True), config)
        print(f"最终执行结果: {result}")
        print(f"最终步骤: {result['step']}")
    else:
        print("用户取消了执行")
else:
    print("图已执行完成")
    
    