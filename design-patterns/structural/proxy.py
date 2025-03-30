# Python设计模式 - 代理模式
# 为其他对象提供一种代理以控制对这个对象的访问

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import time
from datetime import datetime

# 1. 抽象主题
class Subject(ABC):
    """抽象主题接口"""
    
    @abstractmethod
    def request(self, *args, **kwargs) -> str:
        """请求方法"""
        pass

# 2. 真实主题
class RealSubject(Subject):
    """真实主题"""
    
    def request(self, *args, **kwargs) -> str:
        # 模拟耗时操作
        time.sleep(0.1)
        return "RealSubject: 处理请求"

# 3. 代理类型

# 3.1 虚代理
class VirtualProxy(Subject):
    """虚代理 - 延迟加载"""
    
    def __init__(self):
        self._real_subject: Optional[RealSubject] = None
    
    def request(self, *args, **kwargs) -> str:
        # 延迟创建真实对象
        if not self._real_subject:
            print("VirtualProxy: 创建真实对象")
            self._real_subject = RealSubject()
        return self._real_subject.request(*args, **kwargs)

# 3.2 保护代理
class ProtectionProxy(Subject):
    """保护代理 - 访问控制"""
    
    def __init__(self, username: str, password: str):
        self._real_subject = RealSubject()
        self._username = username
        self._password = password
    
    def authenticate(self) -> bool:
        """认证方法"""
        # 模拟认证逻辑
        return self._username == "admin" and self._password == "secret"
    
    def request(self, *args, **kwargs) -> str:
        if self.authenticate():
            return self._real_subject.request(*args, **kwargs)
        return "ProtectionProxy: 访问被拒绝"

# 3.3 远程代理
class RemoteProxy(Subject):
    """远程代理 - 远程访问"""
    
    def __init__(self, service_url: str):
        self._service_url = service_url
    
    def request(self, *args, **kwargs) -> str:
        print(f"RemoteProxy: 连接到远程服务 {self._service_url}")
        # 模拟远程调用
        return f"RemoteProxy: 远程服务返回结果"

# 3.4 缓存代理
class CacheProxy(Subject):
    """缓存代理 - 结果缓存"""
    
    def __init__(self):
        self._real_subject = RealSubject()
        self._cache: Dict[str, tuple] = {}  # 键: 参数, 值: (结果, 时间戳)
        self._cache_duration = 10  # 缓存有效期（秒）
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """检查缓存是否有效"""
        return time.time() - timestamp < self._cache_duration
    
    def request(self, *args, **kwargs) -> str:
        # 生成缓存键
        cache_key = str((args, kwargs))
        
        # 检查缓存
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                print("CacheProxy: 返回缓存结果")
                return result
        
        # 获取新结果
        result = self._real_subject.request(*args, **kwargs)
        self._cache[cache_key] = (result, time.time())
        print("CacheProxy: 缓存新结果")
        return result

# 4. 日志代理
class LoggingProxy(Subject):
    """日志代理 - 记录日志"""
    
    def __init__(self):
        self._real_subject = RealSubject()
        self._log: List[Dict] = []
    
    def request(self, *args, **kwargs) -> str:
        # 记录请求
        start_time = time.time()
        try:
            result = self._real_subject.request(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "error"
        
        # 记录日志
        self._log.append({
            "timestamp": datetime.now().isoformat(),
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "status": status,
            "duration": time.time() - start_time
        })
        
        return result
    
    def get_log(self) -> List[Dict]:
        """获取日志"""
        return self._log

# 5. 使用示例
def proxy_demo():
    print("代理模式示例：")
    
    # 虚代理示例
    print("\n1. 虚代理示例:")
    virtual_proxy = VirtualProxy()
    print("首次调用前...")
    print(virtual_proxy.request())
    print("再次调用...")
    print(virtual_proxy.request())
    
    # 保护代理示例
    print("\n2. 保护代理示例:")
    # 使用错误的凭证
    protection_proxy = ProtectionProxy("user", "wrong")
    print(protection_proxy.request())
    # 使用正确的凭证
    protection_proxy = ProtectionProxy("admin", "secret")
    print(protection_proxy.request())
    
    # 远程代理示例
    print("\n3. 远程代理示例:")
    remote_proxy = RemoteProxy("http://api.example.com")
    print(remote_proxy.request())
    
    # 缓存代理示例
    print("\n4. 缓存代理示例:")
    cache_proxy = CacheProxy()
    print("首次请求:")
    print(cache_proxy.request("param1"))
    print("重复请求:")
    print(cache_proxy.request("param1"))
    
    # 日志代理示例
    print("\n5. 日志代理示例:")
    logging_proxy = LoggingProxy()
    logging_proxy.request("test")
    logging_proxy.request("another_test")
    
    print("\n日志记录:")
    for entry in logging_proxy.get_log():
        print(f"时间: {entry['timestamp']}")
        print(f"参数: {entry['args']}")
        print(f"状态: {entry['status']}")
        print(f"耗时: {entry['duration']:.3f}秒")
        print()

if __name__ == "__main__":
    proxy_demo() 