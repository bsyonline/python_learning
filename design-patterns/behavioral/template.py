# Python设计模式 - 模板方法模式
# 定义一个操作中的算法的骨架，而将一些步骤延迟到子类中

from abc import ABC, abstractmethod

class AbstractClass(ABC):
    def template_method(self):
        self.primitive_operation1()
        self.primitive_operation2()

    @abstractmethod
    def primitive_operation1(self):
        pass

    @abstractmethod
    def primitive_operation2(self):
        pass

class ConcreteClass(AbstractClass):
    def primitive_operation1(self):
        print("ConcreteClass: Primitive Operation 1")

    def primitive_operation2(self):
        print("ConcreteClass: Primitive Operation 2") 