# Python设计模式 - 适配器模式
# 将一个类的接口转换成客户希望的另一个接口，使得原本由于接口不兼容而不能一起工作的类可以一起工作

from abc import ABC, abstractmethod
from typing import Dict, List, Any

# 1. 目标接口
class Target(ABC):
    """客户期待的接口"""
    
    @abstractmethod
    def request(self) -> str:
        pass

# 2. 需要适配的类
class Adaptee:
    """需要适配的类"""
    
    def specific_request(self) -> str:
        return "特殊请求"

# 3. 对象适配器
class ObjectAdapter(Target):
    """对象适配器"""
    
    def __init__(self, adaptee: Adaptee):
        self._adaptee = adaptee
    
    def request(self) -> str:
        # 转换接口
        return f"适配器转换: {self._adaptee.specific_request()}"

# 4. 类适配器
class ClassAdapter(Target, Adaptee):
    """类适配器"""
    
    def request(self) -> str:
        # 直接继承并转换接口
        return f"类适配器转换: {self.specific_request()}"

# 5. 实际应用示例
# 5.1 旧的数据格式处理器
class LegacyDataParser:
    """旧的数据解析器"""
    
    def parse_data(self, data: str) -> Dict[str, Any]:
        # 模拟旧格式解析
        return {"legacy_data": data}
    
    def format_output(self, data: Dict[str, Any]) -> str:
        # 旧格式输出
        return f"Legacy Output: {data}"

# 5.2 新的数据接口
class ModernDataInterface(ABC):
    """现代数据处理接口"""
    
    @abstractmethod
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def generate_output(self, data: List[Dict[str, Any]]) -> str:
        pass

# 5.3 数据处理适配器
class DataAdapter(ModernDataInterface):
    """数据处理适配器"""
    
    def __init__(self, legacy_parser: LegacyDataParser):
        self._parser = legacy_parser
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 将新格式转换为旧格式，处理后再转回新格式
        results = []
        for item in data:
            # 转换为旧格式
            old_format = str(item)
            # 使用旧解析器处理
            processed = self._parser.parse_data(old_format)
            # 转换回新格式
            results.append({"modern_data": processed})
        return results
    
    def generate_output(self, data: List[Dict[str, Any]]) -> str:
        # 将新格式输出转换为旧格式输出
        old_format = {"combined": data}
        return self._parser.format_output(old_format)

# 6. 使用示例
def adapter_demo():
    print("适配器模式示例：")
    
    # 基本适配器示例
    print("\n1. 基本适配器示例:")
    adaptee = Adaptee()
    object_adapter = ObjectAdapter(adaptee)
    class_adapter = ClassAdapter()
    
    print(f"对象适配器: {object_adapter.request()}")
    print(f"类适配器: {class_adapter.request()}")
    
    # 实际应用示例
    print("\n2. 数据处理适配器示例:")
    # 创建旧解析器和适配器
    legacy_parser = LegacyDataParser()
    data_adapter = DataAdapter(legacy_parser)
    
    # 新格式数据
    modern_data = [
        {"id": 1, "name": "项目1"},
        {"id": 2, "name": "项目2"}
    ]
    
    # 使用适配器处理数据
    processed_data = data_adapter.process_data(modern_data)
    output = data_adapter.generate_output(processed_data)
    
    print("处理后的数据:")
    print(processed_data)
    print("\n生成的输出:")
    print(output)

if __name__ == "__main__":
    adapter_demo() 