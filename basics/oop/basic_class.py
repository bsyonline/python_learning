# Python面向对象编程 - 基本类定义和使用

from datetime import datetime

# 1. 基本类定义
class Person:
    """人员类"""
    
    # 类变量
    species = "Homo Sapiens"
    count = 0
    
    # 构造函数
    def __init__(self, name, age):
        # 实例变量
        self.name = name
        self.age = age
        self.created_at = datetime.now()
        Person.count += 1
    
    # 实例方法
    def introduce(self):
        return f"我叫{self.name}，今年{self.age}岁"
    
    def celebrate_birthday(self):
        self.age += 1
        return f"{self.name}过生日，现在{self.age}岁了"
    
    # 类方法
    @classmethod
    def get_species(cls):
        return cls.species
    
    @classmethod
    def get_count(cls):
        return cls.count
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        return age >= 18

# 2. 属性装饰器
class Employee:
    def __init__(self, first_name, last_name, salary):
        self._first_name = first_name
        self._last_name = last_name
        self._salary = salary
    
    # 属性getter
    @property
    def full_name(self):
        return f"{self._first_name} {self._last_name}"
    
    @property
    def salary(self):
        return self._salary
    
    # 属性setter
    @salary.setter
    def salary(self, value):
        if value < 0:
            raise ValueError("薪资不能为负")
        self._salary = value
    
    # 属性deleter
    @salary.deleter
    def salary(self):
        self._salary = 0

# 3. 特殊方法
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # 字符串表示
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    # 运算符重载
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    # 比较方法
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    # 可调用对象
    def __call__(self):
        return (self.x, self.y)

# 4. 使用示例
def basic_class_demo():
    print("基本类使用示例：")
    
    # 创建Person实例
    person1 = Person("张三", 25)
    person2 = Person("李四", 30)
    
    print(person1.introduce())
    print(person2.introduce())
    print(f"目前共有{Person.get_count()}个人")
    print(f"张三是成年人吗？ {Person.is_adult(person1.age)}")
    
    # 使用Employee类
    print("\n员工属性示例：")
    emp = Employee("王", "五", 8000)
    print(f"员工姓名: {emp.full_name}")
    print(f"当前薪资: {emp.salary}")
    
    emp.salary = 10000
    print(f"调薪后: {emp.salary}")
    
    # 使用Point类
    print("\n点操作示例：")
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    
    print(f"p1: {p1}")
    print(f"p2: {p2}")
    print(f"p1 + p2 = {p1 + p2}")
    print(f"p1 == p2: {p1 == p2}")
    print(f"p1的坐标: {p1()}")

# 5. 动态属性
class DynamicClass:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        return self._data.get(name)
    
    def __setattr__(self, name, value):
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value

def dynamic_attributes_demo():
    print("\n动态属性示例：")
    
    obj = DynamicClass()
    obj.name = "动态属性"
    obj.value = 42
    
    print(f"name: {obj.name}")
    print(f"value: {obj.value}")
    print(f"不存在的属性: {obj.unknown}")

if __name__ == "__main__":
    basic_class_demo()
    dynamic_attributes_demo() 