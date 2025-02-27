# Python设计模式 - 建造者模式
# 将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from dataclasses import dataclass

# 1. 产品
@dataclass
class Computer:
    """电脑产品类"""
    cpu: str = ""
    memory: str = ""
    storage: str = ""
    gpu: str = ""
    power: str = ""
    
    def display_specs(self) -> None:
        """显示配置"""
        print("\n电脑配置:")
        print(f"CPU: {self.cpu}")
        print(f"内存: {self.memory}")
        print(f"存储: {self.storage}")
        print(f"显卡: {self.gpu}")
        print(f"电源: {self.power}")

# 2. 抽象建造者
class ComputerBuilder(ABC):
    """电脑建造者抽象类"""
    
    def __init__(self):
        self.computer = Computer()
        self.reset()
    
    def reset(self) -> None:
        """重置产品"""
        self.computer = Computer()
    
    @abstractmethod
    def build_cpu(self) -> None:
        pass
    
    @abstractmethod
    def build_memory(self) -> None:
        pass
    
    @abstractmethod
    def build_storage(self) -> None:
        pass
    
    @abstractmethod
    def build_gpu(self) -> None:
        pass
    
    @abstractmethod
    def build_power(self) -> None:
        pass
    
    def get_result(self) -> Computer:
        """获取构建结果"""
        computer = self.computer
        self.reset()
        return computer

# 3. 具体建造者
class GamingComputerBuilder(ComputerBuilder):
    """游戏电脑建造者"""
    
    def build_cpu(self) -> None:
        self.computer.cpu = "Intel i9-12900K"
    
    def build_memory(self) -> None:
        self.computer.memory = "32GB DDR5"
    
    def build_storage(self) -> None:
        self.computer.storage = "2TB NVMe SSD"
    
    def build_gpu(self) -> None:
        self.computer.gpu = "RTX 4090"
    
    def build_power(self) -> None:
        self.computer.power = "1000W 金牌电源"

class OfficeComputerBuilder(ComputerBuilder):
    """办公电脑建造者"""
    
    def build_cpu(self) -> None:
        self.computer.cpu = "Intel i5-12400"
    
    def build_memory(self) -> None:
        self.computer.memory = "16GB DDR4"
    
    def build_storage(self) -> None:
        self.computer.storage = "512GB SSD"
    
    def build_gpu(self) -> None:
        self.computer.gpu = "集成显卡"
    
    def build_power(self) -> None:
        self.computer.power = "450W 铜牌电源"

# 4. 主管类
class Director:
    """主管类"""
    
    def __init__(self):
        self._builder: Optional[ComputerBuilder] = None
    
    @property
    def builder(self) -> ComputerBuilder:
        if not self._builder:
            raise ValueError("未设置建造者")
        return self._builder
    
    @builder.setter
    def builder(self, builder: ComputerBuilder) -> None:
        self._builder = builder
    
    def build_minimal_computer(self) -> None:
        """构建最小配置"""
        self.builder.build_cpu()
        self.builder.build_memory()
        self.builder.build_storage()
    
    def build_full_computer(self) -> None:
        """构建完整配置"""
        self.builder.build_cpu()
        self.builder.build_memory()
        self.builder.build_storage()
        self.builder.build_gpu()
        self.builder.build_power()

# 5. 自定义建造者
class CustomComputerBuilder:
    """自定义电脑建造者"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self._computer = Computer()
    
    def add_cpu(self, cpu: str) -> 'CustomComputerBuilder':
        self._computer.cpu = cpu
        return self
    
    def add_memory(self, memory: str) -> 'CustomComputerBuilder':
        self._computer.memory = memory
        return self
    
    def add_storage(self, storage: str) -> 'CustomComputerBuilder':
        self._computer.storage = storage
        return self
    
    def add_gpu(self, gpu: str) -> 'CustomComputerBuilder':
        self._computer.gpu = gpu
        return self
    
    def add_power(self, power: str) -> 'CustomComputerBuilder':
        self._computer.power = power
        return self
    
    def build(self) -> Computer:
        """获取构建结果"""
        computer = self._computer
        self.reset()
        return computer

# 6. 使用示例
def builder_demo():
    print("建造者模式示例：")
    
    # 使用主管类构建
    director = Director()
    
    # 构建游戏电脑
    print("\n1. 构建游戏电脑:")
    gaming_builder = GamingComputerBuilder()
    director.builder = gaming_builder
    
    print("最小配置:")
    director.build_minimal_computer()
    gaming_builder.get_result().display_specs()
    
    print("\n完整配置:")
    director.build_full_computer()
    gaming_builder.get_result().display_specs()
    
    # 构建办公电脑
    print("\n2. 构建办公电脑:")
    office_builder = OfficeComputerBuilder()
    director.builder = office_builder
    director.build_full_computer()
    office_builder.get_result().display_specs()
    
    # 使用自定义建造者
    print("\n3. 自定义电脑配置:")
    custom_builder = CustomComputerBuilder()
    custom_computer = (custom_builder
        .add_cpu("AMD Ryzen 7 5800X")
        .add_memory("64GB DDR4")
        .add_storage("1TB SSD + 2TB HDD")
        .add_gpu("RX 6800 XT")
        .add_power("850W 金牌电源")
        .build())
    
    custom_computer.display_specs()

if __name__ == "__main__":
    builder_demo() 