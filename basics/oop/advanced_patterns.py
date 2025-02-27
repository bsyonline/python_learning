# Python面向对象编程 - 高级设计模式示例

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import json

# 1. 责任链模式
class Handler(ABC):
    """处理器抽象类"""
    
    def __init__(self):
        self._next_handler: Optional['Handler'] = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, request: str) -> Optional[str]:
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class AuthenticationHandler(Handler):
    def handle(self, request: str) -> Optional[str]:
        if request.startswith("Auth:"):
            return "认证成功"
        return super().handle(request)

class ValidationHandler(Handler):
    def handle(self, request: str) -> Optional[str]:
        if request.startswith("Valid:"):
            return "验证成功"
        return super().handle(request)

class ProcessingHandler(Handler):
    def handle(self, request: str) -> Optional[str]:
        if request.startswith("Process:"):
            return "处理成功"
        return super().handle(request)

# 2. 状态模式
class State(ABC):
    @abstractmethod
    def handle(self) -> str:
        pass

class OrderState(State):
    def handle(self) -> str:
        return "订单已创建"

class PaymentState(State):
    def handle(self) -> str:
        return "订单已支付"

class ShippingState(State):
    def handle(self) -> str:
        return "订单已发货"

class Order:
    def __init__(self):
        self._state: State = OrderState()
    
    def set_state(self, state: State) -> None:
        self._state = state
    
    def get_status(self) -> str:
        return self._state.handle()

# 3. 代理模式
class Subject(ABC):
    @abstractmethod
    def request(self) -> str:
        pass

class RealSubject(Subject):
    def request(self) -> str:
        return "处理实际请求"

class Proxy(Subject):
    def __init__(self):
        self._real_subject: Optional[RealSubject] = None
    
    def request(self) -> str:
        if not self._real_subject:
            print("代理: 创建实际对象")
            self._real_subject = RealSubject()
        print("代理: 在调用前记录日志")
        result = self._real_subject.request()
        print("代理: 在调用后记录日志")
        return result

# 4. 享元模式
class Character:
    def __init__(self, char: str):
        self.char = char
        # 模拟大量内存占用
        self.font = {"family": "Arial", "size": 12}
        self.color = "black"

class CharacterFactory:
    _characters: Dict[str, Character] = {}
    
    @classmethod
    def get_character(cls, char: str) -> Character:
        if char not in cls._characters:
            cls._characters[char] = Character(char)
        return cls._characters[char]

# 5. 访问者模式
class Element(ABC):
    @abstractmethod
    def accept(self, visitor: 'Visitor') -> None:
        pass

class ConcreteElementA(Element):
    def accept(self, visitor: 'Visitor') -> None:
        visitor.visit_concrete_element_a(self)
    
    def operation_a(self) -> str:
        return "A的操作"

class ConcreteElementB(Element):
    def accept(self, visitor: 'Visitor') -> None:
        visitor.visit_concrete_element_b(self)
    
    def operation_b(self) -> str:
        return "B的操作"

class Visitor(ABC):
    @abstractmethod
    def visit_concrete_element_a(self, element: ConcreteElementA) -> None:
        pass
    
    @abstractmethod
    def visit_concrete_element_b(self, element: ConcreteElementB) -> None:
        pass

class ConcreteVisitor(Visitor):
    def visit_concrete_element_a(self, element: ConcreteElementA) -> None:
        print(f"访问者处理A: {element.operation_a()}")
    
    def visit_concrete_element_b(self, element: ConcreteElementB) -> None:
        print(f"访问者处理B: {element.operation_b()}")

# 6. 备忘录模式
class Memento:
    def __init__(self, state: str):
        self._state = state
        self._date = datetime.now()
    
    def get_state(self) -> str:
        return self._state
    
    def get_date(self) -> datetime:
        return self._date

class Originator:
    def __init__(self):
        self._state = ""
    
    def set_state(self, state: str) -> None:
        print(f"设置状态: {state}")
        self._state = state
    
    def save_to_memento(self) -> Memento:
        print(f"保存状态: {self._state}")
        return Memento(self._state)
    
    def restore_from_memento(self, memento: Memento) -> None:
        self._state = memento.get_state()
        print(f"恢复状态: {self._state}")

class Caretaker:
    def __init__(self):
        self._mementos: List[Memento] = []
    
    def add_memento(self, memento: Memento) -> None:
        self._mementos.append(memento)
    
    def get_memento(self, index: int) -> Memento:
        return self._mementos[index]

# 7. 使用示例
def advanced_patterns_demo():
    print("高级设计模式示例：")
    
    # 责任链模式示例
    print("\n责任链模式示例:")
    auth_handler = AuthenticationHandler()
    valid_handler = ValidationHandler()
    process_handler = ProcessingHandler()
    
    auth_handler.set_next(valid_handler).set_next(process_handler)
    
    print(auth_handler.handle("Auth: 用户登录"))
    print(auth_handler.handle("Valid: 数据验证"))
    print(auth_handler.handle("Process: 业务处理"))
    
    # 状态模式示例
    print("\n状态模式示例:")
    order = Order()
    print(f"初始状态: {order.get_status()}")
    
    order.set_state(PaymentState())
    print(f"支付后状态: {order.get_status()}")
    
    order.set_state(ShippingState())
    print(f"发货后状态: {order.get_status()}")
    
    # 代理模式示例
    print("\n代理模式示例:")
    proxy = Proxy()
    print(proxy.request())
    
    # 享元模式示例
    print("\n享元模式示例:")
    factory = CharacterFactory()
    characters = [factory.get_character(c) for c in "Hello"]
    print(f"创建的字符数: {len(characters)}")
    print(f"实际存储的字符数: {len(CharacterFactory._characters)}")
    
    # 访问者模式示例
    print("\n访问者模式示例:")
    elements = [ConcreteElementA(), ConcreteElementB()]
    visitor = ConcreteVisitor()
    
    for element in elements:
        element.accept(visitor)
    
    # 备忘录模式示例
    print("\n备忘录模式示例:")
    originator = Originator()
    caretaker = Caretaker()
    
    originator.set_state("状态 1")
    caretaker.add_memento(originator.save_to_memento())
    
    originator.set_state("状态 2")
    caretaker.add_memento(originator.save_to_memento())
    
    print("\n恢复到之前的状态:")
    originator.restore_from_memento(caretaker.get_memento(0))

if __name__ == "__main__":
    advanced_patterns_demo() 