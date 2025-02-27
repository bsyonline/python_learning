# Python面向对象编程 - 多态和抽象类

from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable
from datetime import datetime

# 1. 基本多态
class Notification:
    """通知基类"""
    
    def send(self, message: str) -> bool:
        raise NotImplementedError("子类必须实现send方法")

class EmailNotification(Notification):
    """邮件通知"""
    
    def send(self, message: str) -> bool:
        print(f"通过邮件发送: {message}")
        return True

class SMSNotification(Notification):
    """短信通知"""
    
    def send(self, message: str) -> bool:
        print(f"通过短信发送: {message}")
        return True

class PushNotification(Notification):
    """推送通知"""
    
    def send(self, message: str) -> bool:
        print(f"通过推送发送: {message}")
        return True

# 2. 抽象基类
class PaymentMethod(ABC):
    """支付方式抽象基类"""
    
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        """处理支付"""
        pass
    
    @abstractmethod
    def refund(self, amount: float) -> bool:
        """处理退款"""
        pass
    
    @property
    @abstractmethod
    def payment_type(self) -> str:
        """支付类型"""
        pass

class CreditCardPayment(PaymentMethod):
    """信用卡支付"""
    
    def __init__(self, card_number: str, expiry_date: str):
        self.card_number = card_number
        self.expiry_date = expiry_date
    
    def process_payment(self, amount: float) -> bool:
        print(f"使用信用卡 {self.card_number} 支付 {amount}")
        return True
    
    def refund(self, amount: float) -> bool:
        print(f"退款 {amount} 到信用卡 {self.card_number}")
        return True
    
    @property
    def payment_type(self) -> str:
        return "信用卡"

# 3. 协议类（结构化类型）
@runtime_checkable
class Drawable(Protocol):
    """可绘制协议"""
    
    def draw(self) -> None:
        """绘制方法"""
        pass

class Circle:
    """圆形"""
    
    def __init__(self, radius: float):
        self.radius = radius
    
    def draw(self) -> None:
        print(f"绘制半径为 {self.radius} 的圆")

class Rectangle:
    """矩形"""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def draw(self) -> None:
        print(f"绘制 {self.width}x{self.height} 的矩形")

# 4. 鸭子类型
class Logger:
    """日志记录器"""
    
    def log(self, message: str) -> None:
        """记录日志"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] {message}")

class MockLogger:
    """模拟日志记录器"""
    
    def __init__(self):
        self.logs = []
    
    def log(self, message: str) -> None:
        """记录日志到内存"""
        self.logs.append(message)
    
    def get_logs(self) -> List[str]:
        """获取所有日志"""
        return self.logs

# 5. 多态性和依赖注入
class NotificationService:
    """通知服务"""
    
    def __init__(self, notification: Notification):
        self.notification = notification
    
    def send_notification(self, message: str) -> bool:
        return self.notification.send(message)

class PaymentProcessor:
    """支付处理器"""
    
    def __init__(self, payment_method: PaymentMethod):
        self.payment_method = payment_method
    
    def process(self, amount: float) -> bool:
        print(f"使用{self.payment_method.payment_type}处理支付")
        return self.payment_method.process_payment(amount)

class Canvas:
    """画布"""
    
    def draw_shapes(self, shapes: List[Drawable]) -> None:
        """绘制多个形状"""
        for shape in shapes:
            shape.draw()

# 6. 使用示例
def polymorphism_demo():
    print("多态示例：")
    
    # 通知示例
    print("\n不同类型的通知:")
    notifications = [
        EmailNotification(),
        SMSNotification(),
        PushNotification()
    ]
    
    for notification in notifications:
        notification.send("Hello, World!")
    
    # 支付示例
    print("\n支付处理:")
    credit_card = CreditCardPayment("1234-5678-9012-3456", "12/24")
    processor = PaymentProcessor(credit_card)
    processor.process(100.00)
    
    # 形状绘制示例
    print("\n绘制形状:")
    shapes: List[Drawable] = [
        Circle(5),
        Rectangle(10, 20),
        Circle(3)
    ]
    
    canvas = Canvas()
    canvas.draw_shapes(shapes)
    
    # 日志记录示例
    print("\n日志记录:")
    loggers = [Logger(), MockLogger()]
    for logger in loggers:
        logger.log("测试消息")
    
    # 验证是否为可绘制对象
    print("\n类型检查:")
    print(f"Circle 是否可绘制: {isinstance(Circle(1), Drawable)}")
    print(f"Rectangle 是否可绘制: {isinstance(Rectangle(1, 1), Drawable)}")

if __name__ == "__main__":
    polymorphism_demo() 