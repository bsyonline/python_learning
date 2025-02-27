# Python面向对象编程 - 继承和多态

from abc import ABC, abstractmethod
from typing import List

# 1. 基本继承
class Animal:
    """动物基类"""
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def speak(self) -> str:
        return "Some sound"
    
    def introduce(self) -> str:
        return f"我是{self.name}，今年{self.age}岁"

class Dog(Animal):
    """狗类"""
    
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)
        self.breed = breed
    
    def speak(self) -> str:
        return "汪汪!"
    
    def fetch(self) -> str:
        return f"{self.name}在接飞盘"

class Cat(Animal):
    """猫类"""
    
    def __init__(self, name: str, age: int, color: str):
        super().__init__(name, age)
        self.color = color
    
    def speak(self) -> str:
        return "喵喵!"
    
    def climb(self) -> str:
        return f"{self.name}在爬树"

# 2. 多重继承
class Flyable:
    """可飞行接口"""
    
    def fly(self) -> str:
        return "正在飞行"

class Swimmable:
    """可游泳接口"""
    
    def swim(self) -> str:
        return "正在游泳"

class Duck(Animal, Flyable, Swimmable):
    """鸭子类"""
    
    def speak(self) -> str:
        return "嘎嘎!"
    
    def fly(self) -> str:
        return f"{self.name}" + super().fly()
    
    def swim(self) -> str:
        return f"{self.name}" + super().swim()

# 3. 方法解析顺序(MRO)
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B" + super().method()

class C(A):
    def method(self):
        return "C" + super().method()

class D(B, C):
    def method(self):
        return "D" + super().method()

# 4. 抽象基类
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        return 3.14159 * self.radius * self.radius
    
    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# 5. 继承中的属性访问
class Parent:
    def __init__(self):
        self.public_attr = "公共属性"
        self._protected_attr = "保护属性"
        self.__private_attr = "私有属性"
    
    def get_private_attr(self):
        return self.__private_attr

class Child(Parent):
    def __init__(self):
        super().__init__()
        # 可以访问公共和保护属性
        print(f"公共属性: {self.public_attr}")
        print(f"保护属性: {self._protected_attr}")
        # 不能直接访问私有属性
        # print(self.__private_attr)  # 这会引发错误
        print(f"通过方法访问私有属性: {self.get_private_attr()}")

# 6. 使用示例
def inheritance_demo():
    print("继承示例：")
    
    # 创建动物实例
    dog = Dog("旺财", 3, "金毛")
    cat = Cat("咪咪", 2, "橘色")
    duck = Duck("唐老鸭", 1)
    
    # 展示多态
    animals: List[Animal] = [dog, cat, duck]
    for animal in animals:
        print(f"\n{animal.introduce()}")
        print(f"叫声: {animal.speak()}")
    
    # 特定方法调用
    print(f"\n特定行为:")
    print(dog.fetch())
    print(cat.climb())
    print(duck.fly())
    print(duck.swim())
    
    # MRO示例
    print("\nMRO示例:")
    d = D()
    print(f"方法调用顺序: {d.method()}")
    print(f"MRO: {[cls.__name__ for cls in D.__mro__]}")
    
    # 形状示例
    print("\n形状计算:")
    rect = Rectangle(5, 3)
    circle = Circle(2)
    
    shapes: List[Shape] = [rect, circle]
    for shape in shapes:
        print(f"{shape.__class__.__name__}:")
        print(f"面积: {shape.area():.2f}")
        print(f"周长: {shape.perimeter():.2f}")
    
    # 属性访问示例
    print("\n属性访问示例:")
    child = Child()

if __name__ == "__main__":
    inheritance_demo() 