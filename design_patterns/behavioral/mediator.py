# Python设计模式 - 中介者模式
# 用一个中介对象来封装一系列的对象交互，使各对象不需要显式地相互引用，从而使其耦合松散

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

# 1. 中介者接口
class ChatMediator(ABC):
    """聊天中介者接口"""
    
    @abstractmethod
    def send_message(self, message: str, user: 'User') -> None:
        """发送消息"""
        pass
    
    @abstractmethod
    def add_user(self, user: 'User') -> None:
        """添加用户"""
        pass
    
    @abstractmethod
    def remove_user(self, user: 'User') -> None:
        """移除用户"""
        pass

# 2. 用户接口
class User(ABC):
    """用户抽象基类"""
    
    def __init__(self, name: str, mediator: ChatMediator):
        self.name = name
        self.mediator = mediator
        self._messages: List[str] = []
    
    def send(self, message: str) -> None:
        """发送消息"""
        print(f"{self.name} 发送消息: {message}")
        self.mediator.send_message(message, self)
    
    @abstractmethod
    def receive(self, message: str, sender: 'User') -> None:
        """接收消息"""
        pass
    
    def get_messages(self) -> List[str]:
        """获取消息历史"""
        return self._messages

# 3. 具体用户类型
class ChatUser(User):
    """普通聊天用户"""
    
    def receive(self, message: str, sender: 'User') -> None:
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] {sender.name}: {message}"
        self._messages.append(msg)
        print(f"{self.name} 收到消息: {message}")

class AdminUser(User):
    """管理员用户"""
    
    def receive(self, message: str, sender: 'User') -> None:
        msg = f"[ADMIN][{datetime.now().strftime('%H:%M:%S')}] {sender.name}: {message}"
        self._messages.append(msg)
        print(f"管理员 {self.name} 收到消息: {message}")
    
    def broadcast(self, message: str) -> None:
        """广播消息"""
        print(f"管理员 {self.name} 广播: {message}")
        self.mediator.send_message(f"[公告] {message}", self)

# 4. 具体中介者
class ChatRoom(ChatMediator):
    """聊天室中介者"""
    
    def __init__(self):
        self._users: List[User] = []
        self._banned_users: List[User] = []
    
    def add_user(self, user: User) -> None:
        if user not in self._banned_users:
            self._users.append(user)
            print(f"用户 {user.name} 加入聊天室")
    
    def remove_user(self, user: User) -> None:
        if user in self._users:
            self._users.remove(user)
            print(f"用户 {user.name} 离开聊天室")
    
    def send_message(self, message: str, user: User) -> None:
        if user in self._banned_users:
            print(f"用户 {user.name} 已被禁言，无法发送消息")
            return
        
        for u in self._users:
            if u != user:  # 不发送给自己
                u.receive(message, user)
    
    def ban_user(self, user: User) -> None:
        """禁言用户"""
        if user in self._users and user not in self._banned_users:
            self._banned_users.append(user)
            print(f"用户 {user.name} 已被禁言")
    
    def unban_user(self, user: User) -> None:
        """解除禁言"""
        if user in self._banned_users:
            self._banned_users.remove(user)
            print(f"用户 {user.name} 已解除禁言")

# 5. 高级功能：私聊和群组
class PrivateChatMediator(ChatMediator):
    """私聊中介者"""
    
    def __init__(self):
        self._users: Dict[str, User] = {}
    
    def add_user(self, user: User) -> None:
        self._users[user.name] = user
    
    def remove_user(self, user: User) -> None:
        if user.name in self._users:
            del self._users[user.name]
    
    def send_message(self, message: str, user: User, to: str = None) -> None:
        """发送私聊消息"""
        if to and to in self._users:
            self._users[to].receive(f"[私聊] {message}", user)
        else:
            print(f"用户 {to} 不存在")

# 6. 使用示例
def mediator_demo():
    print("中介者模式示例：")
    
    # 创建聊天室
    chat_room = ChatRoom()
    
    # 创建用户
    alice = ChatUser("Alice", chat_room)
    bob = ChatUser("Bob", chat_room)
    charlie = ChatUser("Charlie", chat_room)
    admin = AdminUser("Admin", chat_room)
    
    # 用户加入聊天室
    print("\n1. 用户加入聊天室:")
    chat_room.add_user(alice)
    chat_room.add_user(bob)
    chat_room.add_user(charlie)
    chat_room.add_user(admin)
    
    # 发送消息
    print("\n2. 普通消息交流:")
    alice.send("大家好！")
    bob.send("你好，Alice!")
    
    # 管理员广播
    print("\n3. 管理员广播:")
    admin.broadcast("欢迎来到聊天室！")
    
    # 禁言功能
    print("\n4. 禁言功能测试:")
    chat_room.ban_user(charlie)
    charlie.send("能看到我的消息吗？")  # 此消息不会被发送
    chat_room.unban_user(charlie)
    charlie.send("我又可以说话了！")
    
    # 私聊功能
    print("\n5. 私聊功能测试:")
    private_chat = PrivateChatMediator()
    private_chat.add_user(alice)
    private_chat.add_user(bob)
    
    private_chat.send_message("你好，只有你能看到这条消息", alice, "Bob")
    
    # 显示消息历史
    print("\n6. Bob的消息历史:")
    for message in bob.get_messages():
        print(message)

if __name__ == "__main__":
    mediator_demo() 