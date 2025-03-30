# Python设计模式 - 原型模式
# 用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象

from abc import ABC, abstractmethod
import copy
from typing import Dict, Any, List

# 1. 原型接口
class Prototype(ABC):
    """原型抽象基类"""
    
    @abstractmethod
    def clone(self) -> 'Prototype':
        """克隆方法"""
        pass

# 2. 具体原型
class Document(Prototype):
    """文档类"""
    
    def __init__(self, content: str, formatting: Dict[str, Any]):
        self.content = content
        self.formatting = formatting
        self._cached_objects: List[Any] = []  # 模拟复杂对象缓存
    
    def clone(self) -> 'Document':
        """浅拷贝"""
        return copy.copy(self)
    
    def deep_clone(self) -> 'Document':
        """深拷贝"""
        return copy.deepcopy(self)
    
    def __str__(self) -> str:
        return f"文档内容: {self.content}\n格式设置: {self.formatting}"

class GraphicalShape(Prototype):
    """图形类"""
    
    def __init__(self, shape_type: str, coordinates: List[tuple], color: str):
        self.shape_type = shape_type
        self.coordinates = coordinates
        self.color = color
    
    def clone(self) -> 'GraphicalShape':
        """创建图形副本"""
        return copy.deepcopy(self)
    
    def move(self, dx: int, dy: int) -> None:
        """移动图形"""
        self.coordinates = [(x + dx, y + dy) for x, y in self.coordinates]
    
    def __str__(self) -> str:
        return (f"图形类型: {self.shape_type}\n"
                f"坐标: {self.coordinates}\n"
                f"颜色: {self.color}")

# 3. 原型注册表
class PrototypeRegistry:
    """原型注册表"""
    
    def __init__(self):
        self._prototypes: Dict[str, Prototype] = {}
    
    def register(self, name: str, prototype: Prototype) -> None:
        """注册原型"""
        self._prototypes[name] = prototype
    
    def unregister(self, name: str) -> None:
        """注销原型"""
        del self._prototypes[name]
    
    def clone(self, name: str) -> Prototype:
        """克隆原型"""
        prototype = self._prototypes.get(name)
        if not prototype:
            raise ValueError(f"未找到原型: {name}")
        return prototype.clone()

# 4. 复杂对象示例
class ComplexObject(Prototype):
    """复杂对象类"""
    
    def __init__(self, data: Dict[str, Any], reference: Any = None):
        self.data = data
        self.reference = reference
        self.calculated_value = self._expensive_calculation()
    
    def _expensive_calculation(self) -> float:
        """模拟耗时计算"""
        import time
        time.sleep(0.1)  # 模拟耗时操作
        return sum(range(1000))
    
    def clone(self) -> 'ComplexObject':
        """克隆复杂对象"""
        # 避免重复执行耗时计算
        cloned = copy.deepcopy(self)
        cloned.calculated_value = self.calculated_value
        return cloned
    
    def __str__(self) -> str:
        return f"数据: {self.data}\n计算值: {self.calculated_value}"

# 5. 使用示例
def prototype_demo():
    print("原型模式示例：")
    
    # 文档原型示例
    print("\n1. 文档克隆示例:")
    original_doc = Document(
        content="Hello, World!",
        formatting={"font": "Arial", "size": 12, "color": "black"}
    )
    
    # 浅拷贝
    cloned_doc = original_doc.clone()
    cloned_doc.content = "Hello, Clone!"
    cloned_doc.formatting["color"] = "blue"
    
    print("原始文档:")
    print(original_doc)
    print("\n浅拷贝后的文档:")
    print(cloned_doc)
    
    # 图形原型示例
    print("\n2. 图形克隆示例:")
    original_shape = GraphicalShape(
        shape_type="rectangle",
        coordinates=[(0, 0), (0, 1), (1, 1), (1, 0)],
        color="red"
    )
    
    cloned_shape = original_shape.clone()
    cloned_shape.move(5, 5)
    
    print("原始图形:")
    print(original_shape)
    print("\n移动后的克隆图形:")
    print(cloned_shape)
    
    # 原型注册表示例
    print("\n3. 原型注册表示例:")
    registry = PrototypeRegistry()
    
    # 注册原型
    registry.register("document", Document("模板文档", {"font": "Times"}))
    registry.register("shape", GraphicalShape("circle", [(0, 0)], "green"))
    
    # 克隆注册的原型
    doc_clone = registry.clone("document")
    shape_clone = registry.clone("shape")
    
    print("从注册表克隆的对象:")
    print(doc_clone)
    print(shape_clone)
    
    # 复杂对象克隆示例
    print("\n4. 复杂对象克隆示例:")
    original_complex = ComplexObject({"key": "value"})
    print("原始对象创建完成")
    
    import time
    start_time = time.time()
    cloned_complex = original_complex.clone()
    clone_time = time.time() - start_time
    
    print(f"克隆耗时: {clone_time:.3f}秒")
    print("原始对象:")
    print(original_complex)
    print("\n克隆对象:")
    print(cloned_complex)

if __name__ == "__main__":
    prototype_demo() 