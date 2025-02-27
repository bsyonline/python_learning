# Python设计模式 - 桥接模式
# 将抽象部分与实现部分分离，使它们都可以独立地变化

from abc import ABC, abstractmethod
from typing import List

# 1. 实现部分的接口
class DrawAPI(ABC):
    """绘图API接口"""
    
    @abstractmethod
    def draw_circle(self, x: int, y: int, radius: int) -> None:
        pass
    
    @abstractmethod
    def draw_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        pass

# 2. 具体实现类
class DrawingAPI1(DrawAPI):
    """绘图API实现1"""
    
    def draw_circle(self, x: int, y: int, radius: int) -> None:
        print(f"API1.圆形 -> 中心点({x}, {y}), 半径: {radius}")
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        print(f"API1.矩形 -> 左上角({x}, {y}), 宽: {width}, 高: {height}")

class DrawingAPI2(DrawAPI):
    """绘图API实现2"""
    
    def draw_circle(self, x: int, y: int, radius: int) -> None:
        print(f"API2.圆形 @ 位置({x}, {y}), 大小: {radius}")
    
    def draw_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        print(f"API2.矩形 @ 起点({x}, {y}), 尺寸: {width}x{height}")

# 3. 抽象部分
class Shape(ABC):
    """形状抽象类"""
    
    def __init__(self, drawing_api: DrawAPI):
        self._drawing_api = drawing_api
    
    @abstractmethod
    def draw(self) -> None:
        """绘制形状"""
        pass
    
    @abstractmethod
    def resize(self, factor: float) -> None:
        """调整大小"""
        pass

# 4. 细化抽象
class Circle(Shape):
    """圆形类"""
    
    def __init__(self, x: int, y: int, radius: int, drawing_api: DrawAPI):
        super().__init__(drawing_api)
        self._x = x
        self._y = y
        self._radius = radius
    
    def draw(self) -> None:
        self._drawing_api.draw_circle(self._x, self._y, self._radius)
    
    def resize(self, factor: float) -> None:
        self._radius = int(self._radius * factor)

class Rectangle(Shape):
    """矩形类"""
    
    def __init__(self, x: int, y: int, width: int, height: int, drawing_api: DrawAPI):
        super().__init__(drawing_api)
        self._x = x
        self._y = y
        self._width = width
        self._height = height
    
    def draw(self) -> None:
        self._drawing_api.draw_rectangle(self._x, self._y, self._width, self._height)
    
    def resize(self, factor: float) -> None:
        self._width = int(self._width * factor)
        self._height = int(self._height * factor)

# 5. 复杂示例：绘图应用
class DrawingApplication:
    """绘图应用"""
    
    def __init__(self, drawing_api: DrawAPI):
        self._drawing_api = drawing_api
        self._shapes: List[Shape] = []
    
    def add_shape(self, shape: Shape) -> None:
        """添加形状"""
        self._shapes.append(shape)
    
    def render_all(self) -> None:
        """渲染所有形状"""
        for shape in self._shapes:
            shape.draw()
    
    def resize_all(self, factor: float) -> None:
        """调整所有形状的大小"""
        for shape in self._shapes:
            shape.resize(factor)

# 6. 使用示例
def bridge_demo():
    print("桥接模式示例：")
    
    # 基本示例
    print("\n1. 基本绘图示例:")
    api1 = DrawingAPI1()
    api2 = DrawingAPI2()
    
    circle1 = Circle(100, 100, 50, api1)
    circle2 = Circle(150, 150, 75, api2)
    
    print("原始大小:")
    circle1.draw()
    circle2.draw()
    
    print("\n放大后:")
    circle1.resize(1.5)
    circle2.resize(1.2)
    circle1.draw()
    circle2.draw()
    
    # 复杂应用示例
    print("\n2. 绘图应用示例:")
    # 使用API1创建应用
    app1 = DrawingApplication(api1)
    app1.add_shape(Circle(10, 10, 30, api1))
    app1.add_shape(Rectangle(50, 50, 80, 40, api1))
    
    print("API1绘制:")
    app1.render_all()
    print("\n放大所有形状:")
    app1.resize_all(2.0)
    app1.render_all()
    
    # 使用API2创建应用
    print("\nAPI2绘制:")
    app2 = DrawingApplication(api2)
    app2.add_shape(Circle(20, 20, 40, api2))
    app2.add_shape(Rectangle(60, 60, 100, 50, api2))
    app2.render_all()

if __name__ == "__main__":
    bridge_demo() 