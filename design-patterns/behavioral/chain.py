# Python设计模式 - 责任链模式
# 使多个对象都有机会处理请求，从而避免请求的发送者和接收者之间的耦合关系

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

# 1. 处理器接口
class Handler(ABC):
    """处理器抽象基类"""
    
    def __init__(self):
        self._next_handler: Optional[Handler] = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        """设置下一个处理器"""
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """处理请求"""
        pass

# 2. 具体处理器
class AuthenticationHandler(Handler):
    """身份认证处理器"""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        if not request.get("token"):
            return "AuthenticationHandler: 缺少认证令牌"
        
        if request["token"] != "valid_token":
            return "AuthenticationHandler: 无效的认证令牌"
        
        print("AuthenticationHandler: 认证通过")
        return self._next_handler.handle(request) if self._next_handler else None

class AuthorizationHandler(Handler):
    """权限验证处理器"""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        user_role = request.get("role", "guest")
        required_role = request.get("required_role", "admin")
        
        if user_role != required_role:
            return f"AuthorizationHandler: 权限不足 (需要 {required_role})"
        
        print("AuthorizationHandler: 权限验证通过")
        return self._next_handler.handle(request) if self._next_handler else None

class ValidationHandler(Handler):
    """数据验证处理器"""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        data = request.get("data", {})
        
        if not isinstance(data, dict):
            return "ValidationHandler: 数据格式错误"
        
        if not data.get("name"):
            return "ValidationHandler: 缺少名称字段"
        
        if not data.get("age"):
            return "ValidationHandler: 缺少年龄字段"
        
        print("ValidationHandler: 数据验证通过")
        return self._next_handler.handle(request) if self._next_handler else None

# 3. 日志记录处理器
class LoggingHandler(Handler):
    """日志处理器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 添加控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        self.logger.info(f"处理请求: {request}")
        result = self._next_handler.handle(request) if self._next_handler else None
        if result:
            self.logger.warning(f"请求处理失败: {result}")
        else:
            self.logger.info("请求处理成功")
        return result

# 4. 请求处理器
class RequestProcessor:
    """请求处理器"""
    
    def __init__(self):
        # 创建处理链
        self.handler = LoggingHandler()
        auth_handler = AuthenticationHandler()
        authz_handler = AuthorizationHandler()
        valid_handler = ValidationHandler()
        
        # 组装处理链
        self.handler.set_next(auth_handler).set_next(authz_handler).set_next(valid_handler)
    
    def process_request(self, request: Dict[str, Any]) -> Optional[str]:
        """处理请求"""
        return self.handler.handle(request)

# 5. 使用示例
def chain_demo():
    print("责任链模式示例：")
    
    processor = RequestProcessor()
    
    # 测试各种请求场景
    print("\n1. 缺少令牌的请求:")
    request1 = {
        "data": {"name": "John", "age": 30},
        "role": "admin"
    }
    result1 = processor.process_request(request1)
    print(f"处理结果: {result1}")
    
    print("\n2. 无效令牌的请求:")
    request2 = {
        "token": "invalid_token",
        "data": {"name": "John", "age": 30},
        "role": "admin"
    }
    result2 = processor.process_request(request2)
    print(f"处理结果: {result2}")
    
    print("\n3. 权限不足的请求:")
    request3 = {
        "token": "valid_token",
        "data": {"name": "John", "age": 30},
        "role": "user",
        "required_role": "admin"
    }
    result3 = processor.process_request(request3)
    print(f"处理结果: {result3}")
    
    print("\n4. 数据验证失败的请求:")
    request4 = {
        "token": "valid_token",
        "data": {"name": "John"},  # 缺少age字段
        "role": "admin"
    }
    result4 = processor.process_request(request4)
    print(f"处理结果: {result4}")
    
    print("\n5. 有效的请求:")
    request5 = {
        "token": "valid_token",
        "data": {"name": "John", "age": 30},
        "role": "admin"
    }
    result5 = processor.process_request(request5)
    print(f"处理结果: {result5}")

if __name__ == "__main__":
    chain_demo() 