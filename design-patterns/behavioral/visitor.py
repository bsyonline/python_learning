# Python设计模式 - 访问者模式
# 表示一个作用于某对象结构中的各元素的操作。它使你可以在不改变各元素的类的前提下定义作用于这些元素的新操作

from abc import ABC, abstractmethod

class Element(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass

class ConcreteElementA(Element):
    def __init__(self, name):
        self.name = name

    def accept(self, visitor):
        visitor.visit_concrete_element_a(self)

class ConcreteElementB(Element):
    def __init__(self, name):
        self.name = name

    def accept(self, visitor):
        visitor.visit_concrete_element_b(self)

class Visitor(ABC):
    @abstractmethod
    def visit_concrete_element_a(self, element):
        pass

    @abstractmethod
    def visit_concrete_element_b(self, element):
        pass

class ConcreteVisitor(Visitor):
    def visit_concrete_element_a(self, element):
        print(f"Visitor is processing ConcreteElementA: {element.name}")

    def visit_concrete_element_b(self, element):
        print(f"Visitor is processing ConcreteElementB: {element.name}") 