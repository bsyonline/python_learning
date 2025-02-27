# Python设计模式 - 单例模式
# 确保一个类只有一个实例，并提供一个全局访问点

from typing import Dict, Any, Optional
import threading
import json

# 1. 基本单例模式
class Singleton:
    """基本单例实现"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 确保初始化只执行一次
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.data = {}

# 2. 线程安全的单例模式
class ThreadSafeSingleton:
    """线程安全的单例实现"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# 3. 使用装饰器实现单例
def singleton(cls):
    """单例装饰器"""
    
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Configuration:
    """配置类"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
    
    def load_config(self, filename: str) -> None:
        """从文件加载配置"""
        with open(filename, 'r') as f:
            self._config = json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self._config[key] = value

# 4. 元类实现单例
class SingletonMeta(type):
    """单例元类"""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    """数据库连接类"""
    
    def __init__(self, host: str = "localhost"):
        self.host = host
        self._connected = False
    
    def connect(self) -> None:
        """连接数据库"""
        if not self._connected:
            print(f"连接到数据库: {self.host}")
            self._connected = True
    
    def disconnect(self) -> None:
        """断开数据库连接"""
        if self._connected:
            print(f"断开数据库连接: {self.host}")
            self._connected = False

# 5. 懒加载单例
class LazySingleton:
    """懒加载单例实现"""
    
    _instance: Optional['LazySingleton'] = None
    
    def __init__(self):
        self.data = []
    
    @classmethod
    def get_instance(cls) -> 'LazySingleton':
        """获取实例的方法"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def add_data(self, item: Any) -> None:
        """添加数据"""
        self.data.append(item)
    
    def get_data(self) -> list:
        """获取数据"""
        return self.data

# 6. 使用示例
def singleton_demo():
    print("单例模式示例：")
    
    # 基本单例示例
    print("\n1. 基本单例：")
    s1 = Singleton()
    s2 = Singleton()
    print(f"s1 和 s2 是同一个实例: {s1 is s2}")
    
    # 线程安全单例示例
    print("\n2. 线程安全单例：")
    ts1 = ThreadSafeSingleton()
    ts2 = ThreadSafeSingleton()
    print(f"ts1 和 ts2 是同一个实例: {ts1 is ts2}")
    
    # 装饰器单例示例
    print("\n3. 装饰器单例：")
    config1 = Configuration()
    config2 = Configuration()
    config1.set("debug", True)
    print(f"config2.debug = {config2.get('debug')}")
    print(f"config1 和 config2 是同一个实例: {config1 is config2}")
    
    # 元类单例示例
    print("\n4. 元类单例：")
    db1 = Database("localhost")
    db2 = Database("127.0.0.1")  # 尝试创建不同host的实例
    db1.connect()
    print(f"db1 和 db2 的host相同: {db1.host == db2.host}")
    db1.disconnect()
    
    # 懒加载单例示例
    print("\n5. 懒加载单例：")
    lazy1 = LazySingleton.get_instance()
    lazy1.add_data("数据1")
    
    lazy2 = LazySingleton.get_instance()
    lazy2.add_data("数据2")
    
    print(f"lazy1的数据: {lazy1.get_data()}")
    print(f"lazy2的数据: {lazy2.get_data()}")
    print(f"lazy1 和 lazy2 是同一个实例: {lazy1 is lazy2}")

if __name__ == "__main__":
    singleton_demo() 