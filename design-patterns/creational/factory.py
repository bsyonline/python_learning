# Python设计模式 - 工厂模式
# 定义一个创建对象的接口，让子类决定实例化哪个类

from abc import ABC, abstractmethod
from typing import Dict, Type

# 1. 简单工厂模式
class Animal(ABC):
    """动物抽象类"""
    
    @abstractmethod
    def speak(self) -> str:
        """发出声音"""
        pass
    
    @abstractmethod
    def move(self) -> str:
        """移动方式"""
        pass

class Dog(Animal):
    """狗"""
    
    def speak(self) -> str:
        return "汪汪!"
    
    def move(self) -> str:
        return "跑步"

class Cat(Animal):
    """猫"""
    
    def speak(self) -> str:
        return "喵喵!"
    
    def move(self) -> str:
        return "轻步走"

class Bird(Animal):
    """鸟"""
    
    def speak(self) -> str:
        return "啾啾!"
    
    def move(self) -> str:
        return "飞行"

class SimpleAnimalFactory:
    """简单动物工厂"""
    
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        """创建动物实例"""
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        elif animal_type == "bird":
            return Bird()
        else:
            raise ValueError(f"不支持的动物类型: {animal_type}")

# 2. 工厂方法模式
class AnimalFactory(ABC):
    """动物工厂抽象类"""
    
    @abstractmethod
    def create_animal(self) -> Animal:
        """创建动物的抽象方法"""
        pass

class DogFactory(AnimalFactory):
    """狗工厂"""
    
    def create_animal(self) -> Animal:
        return Dog()

class CatFactory(AnimalFactory):
    """猫工厂"""
    
    def create_animal(self) -> Animal:
        return Cat()

class BirdFactory(AnimalFactory):
    """鸟工厂"""
    
    def create_animal(self) -> Animal:
        return Bird()

# 3. 注册工厂模式
class RegisteredAnimalFactory:
    """注册式动物工厂"""
    
    _animals: Dict[str, Type[Animal]] = {}
    
    @classmethod
    def register(cls, animal_type: str, animal_class: Type[Animal]) -> None:
        """注册新的动物类型"""
        cls._animals[animal_type] = animal_class
    
    @classmethod
    def create_animal(cls, animal_type: str) -> Animal:
        """创建已注册的动物实例"""
        if animal_type not in cls._animals:
            raise ValueError(f"未注册的动物类型: {animal_type}")
        return cls._animals[animal_type]()

# 4. 使用示例
def factory_demo():
    print("工厂模式示例：")
    
    # 简单工厂示例
    print("\n1. 简单工厂模式：")
    simple_factory = SimpleAnimalFactory()
    
    dog = simple_factory.create_animal("dog")
    cat = simple_factory.create_animal("cat")
    bird = simple_factory.create_animal("bird")
    
    print(f"狗说: {dog.speak()}, 移动方式: {dog.move()}")
    print(f"猫说: {cat.speak()}, 移动方式: {cat.move()}")
    print(f"鸟说: {bird.speak()}, 移动方式: {bird.move()}")
    
    # 工厂方法示例
    print("\n2. 工厂方法模式：")
    factories = {
        "dog": DogFactory(),
        "cat": CatFactory(),
        "bird": BirdFactory()
    }
    
    for name, factory in factories.items():
        animal = factory.create_animal()
        print(f"{name}说: {animal.speak()}, 移动方式: {animal.move()}")
    
    # 注册工厂示例
    print("\n3. 注册工厂模式：")
    # 注册动物类
    RegisteredAnimalFactory.register("dog", Dog)
    RegisteredAnimalFactory.register("cat", Cat)
    RegisteredAnimalFactory.register("bird", Bird)
    
    # 创建动物实例
    animals = [
        RegisteredAnimalFactory.create_animal("dog"),
        RegisteredAnimalFactory.create_animal("cat"),
        RegisteredAnimalFactory.create_animal("bird")
    ]
    
    for animal in animals:
        print(f"{animal.__class__.__name__}说: {animal.speak()}, "
              f"移动方式: {animal.move()}")

if __name__ == "__main__":
    factory_demo() 