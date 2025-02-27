# Python面向对象编程 - 设计模式示例

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import copy
import json

# 1. 创建型模式

# 1.1 单例模式
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 1.2 工厂模式
class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def speak(self) -> str:
        return "汪汪!"

class Cat(Animal):
    def speak(self) -> str:
        return "喵喵!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"未知动物类型: {animal_type}")

# 1.3 建造者模式
class Computer:
    def __init__(self):
        self.parts: List[str] = []
    
    def add_part(self, part: str) -> None:
        self.parts.append(part)
    
    def list_parts(self) -> str:
        return f"电脑配置: {', '.join(self.parts)}"

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()
    
    def add_cpu(self) -> 'ComputerBuilder':
        self.computer.add_part("CPU")
        return self
    
    def add_memory(self) -> 'ComputerBuilder':
        self.computer.add_part("内存")
        return self
    
    def add_storage(self) -> 'ComputerBuilder':
        self.computer.add_part("硬盘")
        return self
    
    def build(self) -> Computer:
        return self.computer

# 2. 结构型模式

# 2.1 适配器模式
class OldPrinter:
    def print_old(self, text: str) -> None:
        print(f"旧打印机打印: {text}")

class NewPrinter:
    def print_new(self, text: str, color: str) -> None:
        print(f"新打印机打印: {text} (颜色: {color})")

class PrinterAdapter:
    def __init__(self, printer: Any):
        self.printer = printer
    
    def print(self, text: str, color: str = "黑色") -> None:
        if isinstance(self.printer, OldPrinter):
            self.printer.print_old(text)
        else:
            self.printer.print_new(text, color)

# 2.2 装饰器模式
class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass

class SimpleCoffee(Coffee):
    def cost(self) -> float:
        return 10.0
    
    def description(self) -> str:
        return "简单咖啡"

class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee
    
    def cost(self) -> float:
        return self._coffee.cost()
    
    def description(self) -> str:
        return self._coffee.description()

class MilkDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 5.0
    
    def description(self) -> str:
        return self._coffee.description() + ", 加奶"

# 3. 行为型模式

# 3.1 观察者模式
class Observer(ABC):
    @abstractmethod
    def update(self, message: str) -> None:
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state: str = ""
    
    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state: str) -> None:
        self._state = state
        self.notify()

class MessageObserver(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, message: str) -> None:
        print(f"观察者 {self.name} 收到消息: {message}")

# 3.2 策略模式
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> None:
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount: float) -> None:
        print(f"使用信用卡支付 {amount} 元")

class AlipayPayment(PaymentStrategy):
    def pay(self, amount: float) -> None:
        print(f"使用支付宝支付 {amount} 元")

class PaymentContext:
    def __init__(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def execute_payment(self, amount: float) -> None:
        self._strategy.pay(amount)

# 3.3 命令模式
class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass

class Light:
    def turn_on(self) -> None:
        print("灯打开了")
    
    def turn_off(self) -> None:
        print("灯关闭了")

class LightOnCommand(Command):
    def __init__(self, light: Light):
        self._light = light
    
    def execute(self) -> None:
        self._light.turn_on()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self._light = light
    
    def execute(self) -> None:
        self._light.turn_off()

class RemoteControl:
    def __init__(self):
        self._commands: Dict[str, Command] = {}
    
    def register_command(self, name: str, command: Command) -> None:
        self._commands[name] = command
    
    def execute_command(self, name: str) -> None:
        if name in self._commands:
            self._commands[name].execute()
        else:
            print(f"未知命令: {name}")

# 4. 使用示例
def design_patterns_demo():
    print("设计模式示例：")
    
    # 工厂模式示例
    print("\n工厂模式示例:")
    factory = AnimalFactory()
    dog = factory.create_animal("dog")
    cat = factory.create_animal("cat")
    print(f"狗说: {dog.speak()}")
    print(f"猫说: {cat.speak()}")
    
    # 建造者模式示例
    print("\n建造者模式示例:")
    builder = ComputerBuilder()
    computer = builder.add_cpu().add_memory().add_storage().build()
    print(computer.list_parts())
    
    # 适配器模式示例
    print("\n适配器模式示例:")
    old_printer = OldPrinter()
    new_printer = NewPrinter()
    adapter_old = PrinterAdapter(old_printer)
    adapter_new = PrinterAdapter(new_printer)
    adapter_old.print("Hello")
    adapter_new.print("World", "红色")
    
    # 装饰器模式示例
    print("\n装饰器模式示例:")
    coffee = SimpleCoffee()
    milk_coffee = MilkDecorator(coffee)
    print(f"{milk_coffee.description()}: ¥{milk_coffee.cost()}")
    
    # 观察者模式示例
    print("\n观察者模式示例:")
    subject = Subject()
    observer1 = MessageObserver("观察者1")
    observer2 = MessageObserver("观察者2")
    subject.attach(observer1)
    subject.attach(observer2)
    subject.set_state("新消息!")
    
    # 策略模式示例
    print("\n策略模式示例:")
    context = PaymentContext(CreditCardPayment())
    context.execute_payment(100)
    context = PaymentContext(AlipayPayment())
    context.execute_payment(200)
    
    # 命令模式示例
    print("\n命令模式示例:")
    light = Light()
    remote = RemoteControl()
    remote.register_command("开灯", LightOnCommand(light))
    remote.register_command("关灯", LightOffCommand(light))
    remote.execute_command("开灯")
    remote.execute_command("关灯")

if __name__ == "__main__":
    design_patterns_demo() 