from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
import time


class State(TypedDict):
    input_data: str
    processed_data: str
    validation_result: str
    step: int
    risk_level: str  # 风险等级：low, medium, high
    user_confirmation: bool


def data_processor(state: State) -> State:
    """数据处理节点"""
    print("--- 数据处理节点 ---")
    print(f"处理输入数据: {state['input_data']}")
    
    # 模拟数据处理
    time.sleep(1)
    processed = f"已处理: {state['input_data']}"
    
    # 根据输入数据动态判断风险等级
    if "敏感" in state['input_data'] or "机密" in state['input_data']:
        risk_level = "high"
    elif "重要" in state['input_data']:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    print(f"数据处理完成，风险等级: {risk_level}")
    
    return {
        "processed_data": processed,
        "step": 1,
        "risk_level": risk_level
    }


def risk_assessor(state: State):
    """风险评估节点 - 根据风险等级动态决定是否中断"""
    print("--- 风险评估节点 ---")
    print(f"当前风险等级: {state['risk_level']}")
    
    # 动态中断：只有在高风险时才中断
    if state['risk_level'] == "high":
        print("检测到高风险操作，请求人工确认...")
        # 动态中断，等待人工输入
        user_input = interrupt({
            "message": f"检测到高风险操作: {state['processed_data']}",
            "risk_level": state['risk_level'],
            "prompt": "是否批准此高风险操作？(y/n): "
        })
        
        # 根据用户输入决定是否继续
        if user_input.lower() == "y":
            print("高风险操作已获批准")
            return {
                "validation_result": "approved",
                "user_confirmation": True,
                "step": 2
            }
        else:
            print("高风险操作被拒绝")
            return {
                "validation_result": "rejected",
                "user_confirmation": False,
                "step": 2
            }
    elif state['risk_level'] == "medium":
        print("检测到中等风险操作，自动处理...")
        # 中等风险不需要中断，但会记录
        time.sleep(1)
        return {
            "validation_result": "auto_approved",
            "user_confirmation": False,
            "step": 2
        }
    else:
        print("低风险操作，自动处理...")
        # 低风险直接通过
        time.sleep(1)
        return {
            "validation_result": "auto_approved",
            "user_confirmation": False,
            "step": 2
        }


def data_finalizer(state: State) -> State:
    """数据最终处理节点"""
    print("--- 数据最终处理节点 ---")
    
    if state.get('validation_result') == "rejected":
        print("操作已被拒绝，执行清理工作...")
        return {
            "processed_data": "操作已取消",
            "step": 3
        }
    else:
        print("执行最终数据处理...")
        time.sleep(1)
        final_data = f"最终结果: {state['processed_data']}"
        return {
            "processed_data": final_data,
            "step": 3
        }


# 创建图
builder = StateGraph(State)

# 添加节点
builder.add_node("data_processor", data_processor)
builder.add_node("risk_assessor", risk_assessor)
builder.add_node("data_finalizer", data_finalizer)

# 添加边
builder.add_edge(START, "data_processor")
builder.add_edge("data_processor", "risk_assessor")
builder.add_edge("risk_assessor", "data_finalizer")
builder.add_edge("data_finalizer", END)

# 设置内存检查点
memory = InMemorySaver()

# 编译图（注意：动态中断不需要在compile时指定interrupt_before）
graph = builder.compile(checkpointer=memory)


def run_dynamic_interrupt_example(input_text: str):
    """运行动态中断示例"""
    config = {"configurable": {"thread_id": "dynamic-interrupt-example"}}
    
    print("=== 动态Interrupt示例 ===")
    print(f"输入数据: {input_text}")
    print("启动图执行...")
    
    # 启动图执行
    result = graph.invoke(
        {
            "input_data": input_text, 
            "processed_data": "", 
            "validation_result": "", 
            "step": 0, 
            "risk_level": "low",
            "user_confirmation": False
        }, 
        config
    )
    
    # 检查是否有中断
    snapshot = graph.get_state(config)
    while snapshot.next:  # 如果有下一个节点要执行，说明被中断了
        print(f"\n=== 执行被中断 ===")
        print(f"中断时的下一个节点: {snapshot.next}")
        
        # 获取中断时的信息（这里简化处理）
        # 在实际应用中，你可能需要从状态中获取更多信息来决定如何处理
        
        # 模拟用户输入
        user_response = input("请输入您的决策 (y/n): ")
        
        print("\n=== 恢复执行 ===")
        # 恢复执行
        result = graph.invoke(Command(resume=user_response), config)
        
        # 再次检查状态
        snapshot = graph.get_state(config)
    
    print(f"\n=== 执行完成 ===")
    print(f"最终结果: {result['processed_data']}")
    print(f"最终步骤: {result['step']}")
    if 'user_confirmation' in result:
        print(f"人工确认: {result['user_confirmation']}")


# 测试不同风险级别的输入
if __name__ == "__main__":
    print("测试1: 低风险操作")
    run_dynamic_interrupt_example("普通数据处理")
    
    print("\n" + "="*50 + "\n")
    
    print("测试2: 中等风险操作")
    run_dynamic_interrupt_example("重要数据更新")
    
    print("\n" + "="*50 + "\n")
    
    print("测试3: 高风险操作")
    run_dynamic_interrupt_example("敏感信息删除")