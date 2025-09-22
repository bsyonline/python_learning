from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io

from operator import add

class State(TypedDict):
    # 使用Annotated和add函数来自动合并列表
    idiom_history: Annotated[list, add]

idiom_database = [
    "一心一意",
    "意气风发",
    "发人深省",
    "省吃俭用",
    "用心良苦",
    "苦尽甘来",
    "来日方长",
    "长年累月",
    "月下老人",
    "人山人海"
]

def next_idion(current_idiom: str):
    last_char = current_idiom[-1]
    next_idiom = None
    for idiom in idiom_database:
        if idiom.startswith(last_char) and idiom != current_idiom:
            next_idiom = idiom
            break
    return next_idiom

def node1(state: State) -> State:
    """起始节点，接收用户输入的成语"""
    idiom_history = state.get("idiom_history", [])
    
    if not idiom_history:
        return {"idiom_history": []}
    
    idiom = idiom_history[-1]
    
    if idiom not in idiom_database:
        return state
    print(f"node1:{idiom}")
    return {
        "idiom_history": []
    }

def node2(state: State) -> State:
    """接龙节点，根据当前成语找到下一个成语"""
    idiom_history = state.get("idiom_history", [])
    
    current_idiom = idiom_history[-1]
    
    if not current_idiom:
        return state
    
    next_idiom = next_idion(current_idiom)
    
    if next_idiom:
        history_str = " -> ".join(idiom_history + [next_idiom]) if idiom_history else "无"
        print(f"node2:{history_str}")
        return {
            "idiom_history": [next_idiom]
        }
    else:
        return {
            "idiom_history": ["无法找到接龙的成语，游戏结束。"]
        }

def node3(state: State) -> State:
    """结束节点，输出最终结果"""
    idiom_history = state.get("idiom_history", [])
    
    if not idiom_history:
        return {"idiom_history": ["无成语历史"]}
    
    current_idiom = idiom_history[-1]
    final_idiom = next_idion(current_idiom)
    
    if final_idiom:
        new_idiom = final_idiom
    else:
        new_idiom = "无法找到接龙的成语，游戏结束。"
    
    history_str = " -> ".join(idiom_history + [new_idiom]) if idiom_history else "无"
    print(f"node3: {history_str}")
    return {
        "idiom_history": [new_idiom]
    }

# 构建状态图
graph_builder = StateGraph(State)
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)

# 定义边
graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")
graph_builder.add_edge("node3", END)

# 编译图
graph = graph_builder.compile()

# 运行图
response = graph.invoke({"idiom_history":["一心一意"]})

# 获取图片数据
img_data = graph.get_graph().draw_mermaid_png()

# 使用matplotlib显示图片
img = mpimg.imread(io.BytesIO(img_data), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

