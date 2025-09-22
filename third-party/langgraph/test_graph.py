from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io

class State(TypedDict):
    message: str

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
    idiom = state["message"]
    
    if idiom not in idiom_database:
        return {
            "message": f"输入的成语 '{idiom}' 不在成语库中，请重新输入。"
        }
    next_idiom = next_idion(idiom)
    print(f"node1:{next_idiom}")
    return {
        "message": next_idiom
    }

def node2(state: State) -> State:
    """接龙节点，根据当前成语找到下一个成语"""
    idiom = state.get("message", "")
    
    if not idiom:
        return state  # 如果没有当前成语，则不进行接龙
    
    next_idiom = next_idion(idiom)
    if next_idiom:
        print(f"node2:{next_idiom}")
        return {
            "message": next_idiom
        }
    else:
        return {"message": "无法找到接龙的成语，游戏结束。"}

def node3(state: State) -> State:
    """结束节点，输出最终结果"""
    next_idiom = next_idion(state.get('message', ''))
    print(f"node3:{next_idiom}")
    return {
        "message": next_idiom
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
response = graph.invoke({
    "message": "一心一意"
})

# 获取图片数据
img_data = graph.get_graph().draw_mermaid_png()

# 使用matplotlib显示图片
img = mpimg.imread(io.BytesIO(img_data), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

