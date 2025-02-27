# Python设计模式 - 装饰器模式
# 动态地给一个对象添加一些额外的职责，就增加功能来说，装饰器模式比生成子类更为灵活

from abc import ABC, abstractmethod
from typing import Optional, Callable
import time
import functools

# 1. 基础组件接口
class TextComponent(ABC):
    """文本组件接口"""
    
    @abstractmethod
    def content(self) -> str:
        """获取内容"""
        pass

# 2. 具体组件
class PlainText(TextComponent):
    """普通文本"""
    
    def __init__(self, text: str):
        self._text = text
    
    def content(self) -> str:
        return self._text

# 3. 装饰器基类
class TextDecorator(TextComponent):
    """文本装饰器基类"""
    
    def __init__(self, component: TextComponent):
        self._component = component
    
    def content(self) -> str:
        return self._component.content()

# 4. 具体装饰器
class BoldDecorator(TextDecorator):
    """粗体装饰器"""
    
    def content(self) -> str:
        return f"<b>{super().content()}</b>"

class ItalicDecorator(TextDecorator):
    """斜体装饰器"""
    
    def content(self) -> str:
        return f"<i>{super().content()}</i>"

class UnderlineDecorator(TextDecorator):
    """下划线装饰器"""
    
    def content(self) -> str:
        return f"<u>{super().content()}</u>"

# 5. 函数装饰器示例
def log_execution(func: Callable) -> Callable:
    """记录函数执行时间的装饰器"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    
    return wrapper

def validate_input(min_length: int = 1) -> Callable:
    """输入验证装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, text: str, *args, **kwargs):
            if len(text) < min_length:
                raise ValueError(f"文本长度必须大于等于{min_length}")
            return func(self, text, *args, **kwargs)
        return wrapper
    
    return decorator

# 6. 实际应用示例
class TextProcessor:
    """文本处理器"""
    
    def __init__(self):
        self._text: Optional[str] = None
    
    @validate_input(min_length=3)
    def set_text(self, text: str) -> None:
        """设置文本"""
        self._text = text
    
    @log_execution
    def process_text(self) -> str:
        """处理文本"""
        if not self._text:
            raise ValueError("未设置文本")
        
        # 创建基础组件
        text_component = PlainText(self._text)
        
        # 根据需要添加装饰器
        if len(self._text) > 10:
            text_component = BoldDecorator(text_component)
        if self._text.startswith("Important"):
            text_component = ItalicDecorator(text_component)
        if self._text.endswith("!"):
            text_component = UnderlineDecorator(text_component)
        
        return text_component.content()

# 7. 使用示例
def decorator_demo():
    print("装饰器模式示例：")
    
    # 基本装饰器示例
    print("\n1. 基本文本装饰器:")
    text = PlainText("Hello, World!")
    bold_text = BoldDecorator(text)
    italic_bold_text = ItalicDecorator(bold_text)
    
    print(f"原始文本: {text.content()}")
    print(f"粗体文本: {bold_text.content()}")
    print(f"粗体斜体文本: {italic_bold_text.content()}")
    
    # 组合装饰器示例
    print("\n2. 组合多个装饰器:")
    decorated_text = UnderlineDecorator(
        ItalicDecorator(
            BoldDecorator(
                PlainText("Decorated Text")
            )
        )
    )
    print(f"多重装饰后的文本: {decorated_text.content()}")
    
    # 实际应用示例
    print("\n3. 文本处理器示例:")
    processor = TextProcessor()
    
    try:
        # 测试输入验证
        processor.set_text("ab")  # 会抛出异常
    except ValueError as e:
        print(f"验证错误: {e}")
    
    # 处理不同类型的文本
    texts = [
        "Short text",
        "Important notice",
        "This is a long text that will be bold!",
        "Important warning!"
    ]
    
    for text in texts:
        processor.set_text(text)
        result = processor.process_text()
        print(f"\n输入: {text}")
        print(f"输出: {result}")

if __name__ == "__main__":
    decorator_demo() 