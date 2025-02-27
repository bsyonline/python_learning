# Python基础 - 类和面向对象编程

# 1. 基本类定义
class Person:
    """人员类"""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"我叫{self.name}，今年{self.age}岁"

# 2. 继承
class Student(Person):
    """学生类，继承自Person"""
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def study(self, subject):
        return f"{self.name}正在学习{subject}"

# 3. 私有属性和方法
class BankAccount:
    """银行账户类"""
    def __init__(self, account_number, balance):
        self.__account_number = account_number  # 私有属性
        self.__balance = balance
    
    def get_balance(self):
        return self.__balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

# 4. 类方法和静态方法
class MathHelper:
    """数学工具类"""
    @staticmethod
    def is_even(number):
        return number % 2 == 0
    
    @classmethod
    def create_range(cls, start, end):
        return list(range(start, end))

# 示例使用
if __name__ == "__main__":
    # 测试基本类
    print("基本类示例：")
    person = Person("张三", 25)
    print(person.introduce())
    
    # 测试继承
    print("\n继承示例：")
    student = Student("李四", 18, "2021001")
    print(student.introduce())
    print(student.study("Python"))
    
    # 测试私有属性
    print("\n私有属性示例：")
    account = BankAccount("1234567", 1000)
    print(f"当前余额: {account.get_balance()}")
    account.deposit(500)
    print(f"存款后余额: {account.get_balance()}")
    
    # 测试静态方法和类方法
    print("\n静态方法和类方法示例：")
    print(f"4是偶数吗？ {MathHelper.is_even(4)}")
    print(f"范围列表: {MathHelper.create_range(1, 5)}") 