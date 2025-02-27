# Python面向对象编程 - 组合和聚合

from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

# 1. 基本组合关系
class Engine:
    """发动机类"""
    
    def __init__(self, power: int):
        self.power = power
        self._started = False
    
    def start(self) -> None:
        self._started = True
        print(f"{self.power}马力发动机启动")
    
    def stop(self) -> None:
        self._started = False
        print("发动机停止")
    
    @property
    def is_running(self) -> bool:
        return self._started

class Car:
    """汽车类 - 与发动机是组合关系（强依赖）"""
    
    def __init__(self, model: str, power: int):
        self.model = model
        # 发动机是汽车的组成部分，随汽车创建而创建
        self._engine = Engine(power)
    
    def start(self) -> None:
        print(f"{self.model} 启动")
        self._engine.start()
    
    def stop(self) -> None:
        print(f"{self.model} 停止")
        self._engine.stop()
    
    def status(self) -> str:
        return f"{self.model} - 发动机{'运行中' if self._engine.is_running else '已停止'}"

# 2. 聚合关系
class Student:
    """学生类"""
    
    def __init__(self, name: str, student_id: str):
        self.name = name
        self.student_id = student_id
    
    def study(self) -> None:
        print(f"{self.name} 正在学习")

class Course:
    """课程类"""
    
    def __init__(self, name: str, code: str):
        self.name = name
        self.code = code
        self.students: List[Student] = []  # 聚合关系
    
    def add_student(self, student: Student) -> None:
        self.students.append(student)
        print(f"学生 {student.name} 加入课程 {self.name}")
    
    def remove_student(self, student: Student) -> None:
        if student in self.students:
            self.students.remove(student)
            print(f"学生 {student.name} 退出课程 {self.name}")
    
    def list_students(self) -> None:
        print(f"\n课程 {self.name} 的学生列表:")
        for student in self.students:
            print(f"- {student.name} ({student.student_id})")

# 3. 组合的生命周期管理
@dataclass
class Address:
    """地址类"""
    street: str
    city: str
    country: str
    postal_code: str
    
    def __str__(self) -> str:
        return f"{self.street}, {self.city}, {self.country} {self.postal_code}"

class Person:
    """人员类 - 展示组合的生命周期管理"""
    
    def __init__(self, name: str, street: str, city: str, country: str, postal_code: str):
        self.name = name
        # Address对象的生命周期完全由Person管理
        self._address = Address(street, city, country, postal_code)
        self._created_at = datetime.now()
    
    @property
    def address(self) -> str:
        return str(self._address)
    
    def move(self, new_street: str, new_city: str, new_country: str, new_postal_code: str) -> None:
        # 创建新的Address对象，旧的会被垃圾回收
        self._address = Address(new_street, new_city, new_country, new_postal_code)
        print(f"{self.name} 搬家到 {self._address}")

# 4. 部分-整体关系
class Component:
    """组件基类"""
    
    def __init__(self, name: str):
        self.name = name
        self._parent: Optional['Container'] = None
    
    @property
    def parent(self) -> Optional['Container']:
        return self._parent
    
    @parent.setter
    def parent(self, container: 'Container') -> None:
        self._parent = container
    
    def operation(self) -> str:
        return f"组件 {self.name} 执行操作"

class Container:
    """容器类 - 可以包含多个组件"""
    
    def __init__(self, name: str):
        self.name = name
        self.components: List[Component] = []
    
    def add_component(self, component: Component) -> None:
        component.parent = self
        self.components.append(component)
        print(f"添加组件 {component.name} 到 {self.name}")
    
    def remove_component(self, component: Component) -> None:
        if component in self.components:
            component.parent = None
            self.components.remove(component)
            print(f"从 {self.name} 移除组件 {component.name}")
    
    def operation(self) -> None:
        print(f"\n容器 {self.name} 执行操作:")
        for component in self.components:
            print(f"- {component.operation()}")

# 5. 使用示例
def composition_demo():
    print("组合和聚合示例：")
    
    # 组合示例
    print("\n组合关系示例:")
    car = Car("Tesla Model 3", 300)
    car.start()
    print(car.status())
    car.stop()
    print(car.status())
    
    # 聚合示例
    print("\n聚合关系示例:")
    python_course = Course("Python编程", "CS101")
    student1 = Student("张三", "2021001")
    student2 = Student("李四", "2021002")
    
    python_course.add_student(student1)
    python_course.add_student(student2)
    python_course.list_students()
    python_course.remove_student(student1)
    python_course.list_students()
    
    # 生命周期管理示例
    print("\n生命周期管理示例:")
    person = Person("王五", "中关村街道", "北京", "中国", "100080")
    print(f"{person.name} 的地址: {person.address}")
    person.move("滨海大道", "深圳", "中国", "518000")
    
    # 部分-整体关系示例
    print("\n部分-整体关系示例:")
    system = Container("操作系统")
    cpu = Component("CPU")
    memory = Component("内存")
    disk = Component("硬盘")
    
    system.add_component(cpu)
    system.add_component(memory)
    system.add_component(disk)
    system.operation()
    
    system.remove_component(memory)
    system.operation()

if __name__ == "__main__":
    composition_demo() 