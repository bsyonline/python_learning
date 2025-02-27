# Python面向对象编程 - 封装和访问控制

from typing import Any, Dict, Optional
from dataclasses import dataclass

# 1. 基本封装
class BankAccount:
    """银行账户类"""
    
    def __init__(self, account_number: str, owner: str):
        self._account_number = account_number  # 保护属性
        self._owner = owner                    # 保护属性
        self.__balance = 0.0                   # 私有属性
        self.__transaction_history = []        # 私有属性
    
    # 公共方法
    def deposit(self, amount: float) -> bool:
        """存款方法"""
        if amount > 0:
            self.__balance += amount
            self.__add_transaction("存款", amount)
            return True
        return False
    
    def withdraw(self, amount: float) -> bool:
        """取款方法"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__add_transaction("取款", amount)
            return True
        return False
    
    def get_balance(self) -> float:
        """获取余额"""
        return self.__balance
    
    def get_transaction_history(self) -> list:
        """获取交易历史"""
        return self.__transaction_history.copy()
    
    # 私有方法
    def __add_transaction(self, type_: str, amount: float):
        """添加交易记录"""
        self.__transaction_history.append({
            "type": type_,
            "amount": amount,
            "balance": self.__balance
        })

# 2. 属性装饰器的高级用法
class Temperature:
    """温度类"""
    
    def __init__(self, celsius: float = 0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        """摄氏度"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float):
        if value < -273.15:  # 绝对零度
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        """华氏度"""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self) -> float:
        """开尔文"""
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, value: float):
        self.celsius = value - 273.15

# 3. 数据类封装
@dataclass
class Person:
    """人员数据类"""
    
    name: str
    age: int
    email: str
    _salary: float = 0.0  # 保护属性
    
    def __post_init__(self):
        """数据验证"""
        if self.age < 0:
            raise ValueError("年龄不能为负")
        if not '@' in self.email:
            raise ValueError("邮箱格式不正确")
    
    @property
    def salary(self) -> float:
        return self._salary
    
    @salary.setter
    def salary(self, value: float):
        if value < 0:
            raise ValueError("工资不能为负")
        self._salary = value

# 4. 命名空间封装
class NamespaceDemo:
    """命名空间示例"""
    
    def __init__(self):
        self.__dict = {}  # 私有字典存储属性
    
    def __getattr__(self, name: str) -> Any:
        """获取属性"""
        return self.__dict.get(name)
    
    def __setattr__(self, name: str, value: Any):
        """设置属性"""
        if name == '_NamespaceDemo__dict':
            super().__setattr__(name, value)
        else:
            self.__dict[name] = value
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """获取所有属性"""
        return self.__dict.copy()

# 5. 上下文管理器封装
class DatabaseConnection:
    """数据库连接类"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.__connection = None
    
    def __enter__(self):
        """进入上下文"""
        print(f"连接到数据库: {self.connection_string}")
        self.__connection = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.__connection:
            print("关闭数据库连接")
            self.__connection = None
        return False  # 不处理异常
    
    def execute(self, query: str) -> Optional[str]:
        """执行查询"""
        if not self.__connection:
            raise RuntimeError("未连接到数据库")
        return f"执行查询: {query}"

# 6. 使用示例
def encapsulation_demo():
    print("封装示例：")
    
    # 银行账户示例
    print("\n银行账户操作:")
    account = BankAccount("1234567890", "张三")
    account.deposit(1000)
    account.withdraw(500)
    print(f"当前余额: {account.get_balance()}")
    print("交易历史:", account.get_transaction_history())
    
    # 温度转换示例
    print("\n温度转换:")
    temp = Temperature(25)
    print(f"摄氏度: {temp.celsius}")
    print(f"华氏度: {temp.fahrenheit}")
    print(f"开尔文: {temp.kelvin}")
    
    temp.fahrenheit = 100
    print(f"设置华氏度后的摄氏度: {temp.celsius}")
    
    # 数据类示例
    print("\n数据类示例:")
    person = Person("李四", 30, "lisi@example.com")
    person.salary = 8000
    print(f"姓名: {person.name}, 年龄: {person.age}, 工资: {person.salary}")
    
    # 命名空间示例
    print("\n命名空间示例:")
    ns = NamespaceDemo()
    ns.name = "动态属性"
    ns.value = 42
    print("所有属性:", ns.get_all_attributes())
    
    # 上下文管理器示例
    print("\n上下文管理器示例:")
    with DatabaseConnection("mysql://localhost/db") as db:
        result = db.execute("SELECT * FROM users")
        print(result)

if __name__ == "__main__":
    encapsulation_demo() 