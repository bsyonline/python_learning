# Python面向对象编程 - 元类和类装饰器

from typing import Type, Dict, Any, Callable
import functools
import time

# 1. 基本元类
class SingletonMeta(type):
    """单例模式元类"""
    
    _instances: Dict[Type, Any] = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# 使用元类的类
class Database(metaclass=SingletonMeta):
    """数据库连接类"""
    
    def __init__(self, host: str = "localhost"):
        self.host = host
        print(f"创建到 {host} 的数据库连接")
    
    def query(self, sql: str) -> str:
        return f"执行查询: {sql}"

# 2. 类装饰器
def singleton(cls):
    """单例模式装饰器"""
    
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance
    
    wrapper_singleton.instance = None
    return wrapper_singleton

@singleton
class Configuration:
    """配置类"""
    
    def __init__(self):
        self._config = {}
    
    def set(self, key: str, value: Any) -> None:
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

# 3. 方法装饰器
def log_calls(func):
    """记录方法调用的装饰器"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用方法 {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"方法 {func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    
    return wrapper

# 4. 属性验证元类
class ValidatorMeta(type):
    """属性验证元类"""
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # 遍历所有属性
        for key, value in namespace.items():
            if isinstance(value, property):
                # 为属性添加验证
                namespace[key] = mcs.validate_property(value)
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def validate_property(prop):
        """添加属性验证"""
        
        @property
        def wrapper(self):
            value = prop.fget(self)
            if value is None:
                raise ValueError(f"属性 {prop.fget.__name__} 不能为None")
            return value
        
        return wrapper

# 5. 自动注册元类
class PluginMeta(type):
    """插件注册元类"""
    
    plugins = {}
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        cls = super().__new__(mcs, name, bases, namespace)
        if hasattr(cls, 'plugin_name'):
            mcs.plugins[cls.plugin_name] = cls
        return cls

# 6. 使用示例
def metaclass_demo():
    print("元类和装饰器示例：")
    
    # 单例元类示例
    print("\n单例元类示例:")
    db1 = Database("localhost")
    db2 = Database("127.0.0.1")
    print(f"db1 和 db2 是同一个实例: {db1 is db2}")
    
    # 单例装饰器示例
    print("\n单例装饰器示例:")
    config1 = Configuration()
    config2 = Configuration()
    config1.set("debug", True)
    print(f"config2.debug = {config2.get('debug')}")
    print(f"config1 和 config2 是同一个实例: {config1 is config2}")
    
    # 方法装饰器示例
    class Calculator:
        @log_calls
        def complex_calculation(self, n: int) -> int:
            time.sleep(0.1)  # 模拟复杂计算
            return n * n
    
    print("\n方法装饰器示例:")
    calc = Calculator()
    result = calc.complex_calculation(5)
    print(f"计算结果: {result}")
    
    # 属性验证元类示例
    class User(metaclass=ValidatorMeta):
        def __init__(self, name: str):
            self._name = name
        
        @property
        def name(self) -> str:
            return self._name
        
        @name.setter
        def name(self, value: str):
            self._name = value
    
    print("\n属性验证示例:")
    user = User("张三")
    print(f"用户名: {user.name}")
    try:
        user.name = None
    except ValueError as e:
        print(f"验证错误: {e}")
    
    # 插件注册示例
    class Plugin(metaclass=PluginMeta):
        pass
    
    class AudioPlugin(Plugin):
        plugin_name = "audio"
        
        def process(self):
            print("处理音频")
    
    class VideoPlugin(Plugin):
        plugin_name = "video"
        
        def process(self):
            print("处理视频")
    
    print("\n插件注册示例:")
    print(f"已注册的插件: {PluginMeta.plugins.keys()}")
    for name, plugin_cls in PluginMeta.plugins.items():
        plugin = plugin_cls()
        print(f"执行插件 {name}:")
        plugin.process()

if __name__ == "__main__":
    metaclass_demo() 