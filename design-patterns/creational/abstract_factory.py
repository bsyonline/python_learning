# Python设计模式 - 抽象工厂模式
# 提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类

from abc import ABC, abstractmethod
from typing import Dict, Type

# 1. 抽象产品
class Button(ABC):
    """按钮抽象类"""
    
    @abstractmethod
    def paint(self) -> str:
        """绘制按钮"""
        pass

class Window(ABC):
    """窗口抽象类"""
    
    @abstractmethod
    def render(self) -> str:
        """渲染窗口"""
        pass

class ScrollBar(ABC):
    """滚动条抽象类"""
    
    @abstractmethod
    def scroll(self) -> str:
        """滚动"""
        pass

# 2. 具体产品 - Windows风格
class WindowsButton(Button):
    def paint(self) -> str:
        return "渲染Windows风格按钮"

class WindowsWindow(Window):
    def render(self) -> str:
        return "渲染Windows风格窗口"

class WindowsScrollBar(ScrollBar):
    def scroll(self) -> str:
        return "使用Windows风格滚动条"

# 3. 具体产品 - MacOS风格
class MacOSButton(Button):
    def paint(self) -> str:
        return "渲染MacOS风格按钮"

class MacOSWindow(Window):
    def render(self) -> str:
        return "渲染MacOS风格窗口"

class MacOSScrollBar(ScrollBar):
    def scroll(self) -> str:
        return "使用MacOS风格滚动条"

# 4. 抽象工厂
class GUIFactory(ABC):
    """GUI工厂抽象类"""
    
    @abstractmethod
    def create_button(self) -> Button:
        """创建按钮"""
        pass
    
    @abstractmethod
    def create_window(self) -> Window:
        """创建窗口"""
        pass
    
    @abstractmethod
    def create_scrollbar(self) -> ScrollBar:
        """创建滚动条"""
        pass

# 5. 具体工厂
class WindowsFactory(GUIFactory):
    """Windows GUI工厂"""
    
    def create_button(self) -> Button:
        return WindowsButton()
    
    def create_window(self) -> Window:
        return WindowsWindow()
    
    def create_scrollbar(self) -> ScrollBar:
        return WindowsScrollBar()

class MacOSFactory(GUIFactory):
    """MacOS GUI工厂"""
    
    def create_button(self) -> Button:
        return MacOSButton()
    
    def create_window(self) -> Window:
        return MacOSWindow()
    
    def create_scrollbar(self) -> ScrollBar:
        return MacOSScrollBar()

# 6. 应用程序类
class Application:
    """应用程序类"""
    
    def __init__(self, factory: GUIFactory):
        self.factory = factory
        self.button = factory.create_button()
        self.window = factory.create_window()
        self.scrollbar = factory.create_scrollbar()
    
    def create_ui(self) -> None:
        """创建UI"""
        print(self.window.render())
        print(self.button.paint())
        print(self.scrollbar.scroll())

# 7. 工厂提供者
class GUIFactoryProvider:
    """GUI工厂提供者"""
    
    _factories: Dict[str, Type[GUIFactory]] = {
        "windows": WindowsFactory,
        "macos": MacOSFactory
    }
    
    @classmethod
    def get_factory(cls, os_type: str) -> GUIFactory:
        """获取对应操作系统的工厂"""
        factory_class = cls._factories.get(os_type.lower())
        if not factory_class:
            raise ValueError(f"不支持的操作系统类型: {os_type}")
        return factory_class()

# 8. 使用示例
def abstract_factory_demo():
    print("抽象工厂模式示例：")
    
    # Windows应用程序
    print("\n1. 创建Windows风格UI:")
    windows_factory = GUIFactoryProvider.get_factory("windows")
    windows_app = Application(windows_factory)
    windows_app.create_ui()
    
    # MacOS应用程序
    print("\n2. 创建MacOS风格UI:")
    macos_factory = GUIFactoryProvider.get_factory("macos")
    macos_app = Application(macos_factory)
    macos_app.create_ui()
    
    # 动态创建
    print("\n3. 根据配置创建UI:")
    os_types = ["windows", "macos"]
    for os_type in os_types:
        print(f"\n{os_type.upper()}风格:")
        factory = GUIFactoryProvider.get_factory(os_type)
        app = Application(factory)
        app.create_ui()

if __name__ == "__main__":
    abstract_factory_demo() 