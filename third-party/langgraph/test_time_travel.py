from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(Dict):
    input: str
    thoughts: List[str]
    decisions: List[str]
    output: str
    step: int

def planning_node(state: State) -> State:
    """规划节点"""
    print("--- 执行规划节点 ---")
    thoughts = state.get("thoughts", [])
    thoughts.append(f"分析输入: {state.get('input', '')}")
    thoughts.append("制定执行计划")
    
    return {
        "thoughts": thoughts,
        "step": 1
    }

def research_node(state: State) -> State:
    """研究节点"""
    print("--- 执行研究节点 ---")
    thoughts = state.get("thoughts", [])
    thoughts.append("收集相关信息")
    thoughts.append("分析数据")
    
    return {
        "thoughts": thoughts,
        "step": 2
    }

def decision_node(state: State) -> State:
    """决策节点"""
    print("--- 执行决策节点 ---")
    thoughts = state.get("thoughts", [])
    thoughts.append("评估选项")
    
    # 根据输入内容做出不同的决策
    input_text = state.get('input', '').lower()
    if '效率' in input_text:
        decision = "优化工作流程"
    elif '问题' in input_text:
        decision = "执行错误恢复程序"
    elif '资源' in input_text:
        decision = "采用最佳实践"
    else:
        decision = "通用解决方案"
    
    decisions = state.get("decisions", [])
    decisions.append(decision)
    
    return {
        "thoughts": thoughts,
        "decisions": decisions,
        "step": 3
    }

def execution_node(state: State) -> State:
    """执行节点"""
    print("--- 执行节点 ---")
    thoughts = state.get("thoughts", [])
    thoughts.append("执行决策")
    
    last_decision = state.get("decisions", [])[-1] if state.get("decisions") else "默认策略"
    output = f"针对 '{state.get('input', '')}'，我们决定采取 '{last_decision}' 策略。这将帮助我们实现目标。"
    
    return {
        "thoughts": thoughts,
        "output": output,
        "step": 4
    }

workflow = StateGraph(State)
    
# 添加节点
workflow.add_node("planning", planning_node)
workflow.add_node("research", research_node)
workflow.add_node("decision", decision_node)
workflow.add_node("execution", execution_node)

# 添加边
workflow.add_edge(START, "planning")
workflow.add_edge("planning", "research")
workflow.add_edge("research", "decision")
workflow.add_edge("decision", "execution")
workflow.add_edge("execution", END)

# 设置检查点
memory = InMemorySaver()
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 3}}

first_input = {"input": "nihao", "thoughts": [], "decisions": [], "output": "", "step": 0}

print("-" * 20 + " first " + "-" * 20)
first_result = graph.invoke(first_input, config)

print(f"最终输出: {first_result['output']}")
print(f"完整输出: {first_result}")

first_history = list(graph.get_state_history(config))
print(f"\n生成了 {len(first_history)} 个历史检查点")

for i, checkpoint in enumerate(first_history):
    print(f"检查点 {i}: values: {checkpoint.values}")
    print()

print("-" * 20 + " second " + "-" * 20)

# 选择一个历史点进行分支
historical_checkpoint = first_history[2]
print(f"选择的历史点: index: {2},  values: {historical_checkpoint.values}")
graph.update_state(config, historical_checkpoint.values)

# 创建一个新的状态，包含必要的字段
new_state = {
    "input": "chileme",
    "thoughts": graph.get_state(config).values.get("thoughts", []),
    "decisions": graph.get_state(config).values.get("decisions", []),
    "output": "",
    "step": graph.get_state(config).values.get("step", 0)
}

second_result = graph.invoke(new_state, config)
print(f"最终输出: {second_result['output']}")
print(f"完整输出: {second_result}")

second_history = list(graph.get_state_history(config))
print(f"\n生成了 {len(second_history)} 个历史检查点")

for i, checkpoint in enumerate(second_history):
    print(f"检查点 {i}: values: {checkpoint.values}")
    print()

print("-" * 20 + " third " + "-" * 20)
