import time
from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

class State(Dict):
    input: str
    thoughts: List[str]
    decisions: List[str]
    output: str
    step: int


def planning_node(state: State) -> State:
    """规划节点：分析输入并制定计划"""
    print("--- 执行规划节点 ---")
    print(f"输入: {state.get('input', '')}")
    
    # 模拟一些处理时间
    time.sleep(0.5)
    
    thoughts = state.get("thoughts", [])
    thoughts.append(f"分析输入: {state.get('input', '')}")
    thoughts.append("制定执行计划")
    
    return {
        "thoughts": thoughts,
        "step": 1
    }


def research_node(state: State) -> State:
    """研究节点：收集相关信息"""
    print("--- 执行研究节点 ---")
    print(f"当前思考: {state.get('thoughts', [])}")
    
    # 模拟一些处理时间
    time.sleep(0.5)
    
    thoughts = state.get("thoughts", [])
    thoughts.append("收集相关信息")
    thoughts.append("分析数据")
    
    return {
        "thoughts": thoughts,
        "step": 2
    }


def decision_node(state: State) -> State:
    """决策节点：基于研究做出决策"""
    print("--- 执行决策节点 ---")
    print(f"当前思考: {state.get('thoughts', [])}")
    
    # 模拟一些处理时间
    time.sleep(0.5)
    
    thoughts = state.get("thoughts", [])
    thoughts.append("评估选项")
    
    decisions = state.get("decisions", [])
    
    # 基于输入内容做出不同的决策
    input_text = state.get("input", "").lower()
    if "生产力" in input_text:
        decisions.append("实施敏捷开发方法")
    elif "效率" in input_text:
        decisions.append("优化工作流程")
    else:
        decisions.append("采用最佳实践")
    
    return {
        "thoughts": thoughts,
        "decisions": decisions,
        "step": 3
    }


def execution_node(state: State) -> State:
    """执行节点：执行决策并生成输出"""
    print("--- 执行节点 ---")
    print(f"决策: {state.get('decisions', [])}")
    
    # 模拟一些处理时间
    time.sleep(0.5)
    
    thoughts = state.get("thoughts", [])
    thoughts.append("执行决策")
    
    # 基于决策生成不同的输出
    last_decision = state.get("decisions", [])[-1] if state.get("decisions", []) else "默认方案"
    output = f"针对 '{state.get('input', '')}'，我们决定采取 '{last_decision}' 策略。这将帮助我们实现目标。"
    
    return {
        "thoughts": thoughts,
        "output": output,
        "step": 4
    }


def create_graph():
    """创建并返回LangGraph工作流"""
    # 创建状态图
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
    
    # 编译图
    app = workflow.compile(checkpointer=memory)
    
    return app


def demonstrate_time_travel_basics():
    """演示时间旅行的基本功能"""
    print("=== LangGraph 时间旅行功能演示 ===\n")
    
    # 创建图
    app = create_graph()
    
    # 配置
    config = {"configurable": {"thread_id": str(uuid4())}}
    
    # 初始输入
    initial_input = {"input": "我们应该如何提高团队的生产力？", "thoughts": [], "decisions": [], "output": "", "step": 0}
    
    print("1. 首次执行图:")
    print("-" * 40)
    
    # 首次执行
    result = app.invoke(initial_input, config)
    print(f"\n最终输出: {result['output']}")
    print(f"执行步骤: {result['step']}\n")
    
    # 获取历史状态
    print("2. 获取执行历史:")
    print("-" * 40)
    
    # 获取状态历史
    history = list(app.get_state_history(config))
    print(f"历史记录数量: {len(history)}")
    
    # 显示历史记录
    for i, checkpoint in enumerate(history):
        print(f"检查点 {i}: 步骤={checkpoint.values.get('step', 'N/A')}, "
              f"下一个节点={checkpoint.next}")
    
    return app, config, history, result


def demonstrate_branching_from_history(app, config, history, original_result):
    """演示从历史状态分支并探索不同路径"""
    print(f"\n3. 从历史状态分支并探索不同路径:")
    print("-" * 40)
    
    # 选择在决策节点之前的状态（步骤2，研究节点之后）
    historical_checkpoint = history[2]
    print(f"分支点: 步骤={historical_checkpoint.values.get('step', 'N/A')}")
    print(f"当时的思考: {historical_checkpoint.values.get('thoughts', [])}")
    
    # 创建新的配置用于分支
    branch_config = {"configurable": {"thread_id": str(uuid4())}}
    
    # 在分支点恢复状态
    # 使用update_state方法将历史状态恢复到新的配置中
    app.update_state(branch_config, historical_checkpoint.values)
    
    # 修改状态以探索不同的路径
    # 在这个例子中，我们修改输入来影响决策节点的行为
    modified_input = "我们应该如何提高工作效率？"
    app.update_state(branch_config, {"input": modified_input})
    
    print(f"\n修改输入为: {modified_input}")
    print("从修改后的状态继续执行...")
    
    # 继续执行分支路径
    branch_result = app.invoke(None, branch_config)
    print(f"\n分支路径的输出: {branch_result['output']}")
    print(f"执行步骤: {branch_result['step']}")
    
    # 比较结果
    print("\n4. 结果比较:")
    print("-" * 40)
    print(f"原始路径输出: {original_result['output']}")
    print(f"分支路径输出: {branch_result['output']}")
    print(f"\n决策比较:")
    print(f"- 原始决策: {original_result.get('decisions', ['N/A'])[-1] if original_result.get('decisions') else 'N/A'}")
    print(f"- 分支决策: {branch_result.get('decisions', ['N/A'])[-1] if branch_result.get('decisions') else 'N/A'}")


def explore_multiple_alternatives(app, config, history):
    """探索多个替代路径"""
    print("\n\n=== 探索多个替代路径 ===\n")
    
    # 探索三种不同的分支
    branches = [
        "我们应该如何提高工作效率？",
        "我们应该采用什么最佳实践？",
        "我们应该如何优化资源分配？"
    ]
    
    branch_results = []
    
    for i, branch_input in enumerate(branches):
        print(f"\n--- 探索分支 {i+1}: {branch_input} ---")
        
        # 创建新的配置用于分支
        branch_config = {"configurable": {"thread_id": str(uuid4())}}
        
        # 在决策节点之前的状态恢复
        historical_checkpoint = history[2]  # 步骤2的状态
        app.update_state(branch_config, historical_checkpoint.values)
        
        # 修改输入
        app.update_state(branch_config, {"input": branch_input})
        
        # 继续执行
        branch_result = app.invoke(None, branch_config)
        branch_results.append((branch_input, branch_result))
        
        print(f"分支输出: {branch_result['output']}")
        print(f"决策: {branch_result.get('decisions', ['N/A'])[-1] if branch_result.get('decisions') else 'N/A'}")
    
    # 总结比较
    print("\n=== 分支路径总结 ===")
    print("-" * 40)
    for branch_input, branch_result in branch_results:
        decision = branch_result.get('decisions', ['N/A'])[-1] if branch_result.get('decisions') else 'N/A'
        print(f"输入: {branch_input}")
        print(f"决策: {decision}")
        print(f"输出: {branch_result['output'][:50]}...")
        print()


def main():
    """主函数"""
    # 演示基本时间旅行功能
    app, config, history, original_result = demonstrate_time_travel_basics()

    # 演示从历史状态分支
    demonstrate_branching_from_history(app, config, history, original_result)
    
    # 探索多个替代路径
    explore_multiple_alternatives(app, config, history)


if __name__ == "__main__":
    main()