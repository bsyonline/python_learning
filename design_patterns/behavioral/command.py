# Python设计模式 - 命令模式
# 将一个请求封装为一个对象，从而使你可用不同的请求对客户进行参数化，对请求排队或记录请求日志，以及支持可撤销的操作

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

# 1. 命令接口
class Command(ABC):
    """命令抽象基类"""
    
    @abstractmethod
    def execute(self) -> None:
        """执行命令"""
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """撤销命令"""
        pass

# 2. 接收者
class Document:
    """文档类 - 命令接收者"""
    
    def __init__(self):
        self.content = ""
        self._clipboard = ""
    
    def insert_text(self, text: str, position: int = -1) -> None:
        """插入文本"""
        if position == -1:
            self.content += text
        else:
            self.content = self.content[:position] + text + self.content[position:]
    
    def delete_text(self, start: int, end: int) -> str:
        """删除文本"""
        deleted_text = self.content[start:end]
        self.content = self.content[:start] + self.content[end:]
        return deleted_text
    
    def copy_text(self, start: int, end: int) -> None:
        """复制文本"""
        self._clipboard = self.content[start:end]
    
    def paste_text(self, position: int = -1) -> None:
        """粘贴文本"""
        self.insert_text(self._clipboard, position)
    
    def get_content(self) -> str:
        """获取内容"""
        return self.content

# 3. 具体命令
class InsertCommand(Command):
    """插入命令"""
    
    def __init__(self, document: Document, text: str, position: int = -1):
        self.document = document
        self.text = text
        self.position = position
        self.length = len(text)
    
    def execute(self) -> None:
        self.document.insert_text(self.text, self.position)
    
    def undo(self) -> None:
        if self.position == -1:
            self.document.delete_text(len(self.document.content) - self.length, 
                                    len(self.document.content))
        else:
            self.document.delete_text(self.position, self.position + self.length)

class DeleteCommand(Command):
    """删除命令"""
    
    def __init__(self, document: Document, start: int, end: int):
        self.document = document
        self.start = start
        self.end = end
        self.deleted_text = ""
    
    def execute(self) -> None:
        self.deleted_text = self.document.delete_text(self.start, self.end)
    
    def undo(self) -> None:
        self.document.insert_text(self.deleted_text, self.start)

class CopyPasteCommand(Command):
    """复制粘贴命令"""
    
    def __init__(self, document: Document, copy_start: int, copy_end: int, 
                 paste_position: int = -1):
        self.document = document
        self.copy_start = copy_start
        self.copy_end = copy_end
        self.paste_position = paste_position
        self.paste_length = copy_end - copy_start
    
    def execute(self) -> None:
        self.document.copy_text(self.copy_start, self.copy_end)
        self.document.paste_text(self.paste_position)
    
    def undo(self) -> None:
        if self.paste_position == -1:
            self.document.delete_text(len(self.document.content) - self.paste_length,
                                    len(self.document.content))
        else:
            self.document.delete_text(self.paste_position,
                                    self.paste_position + self.paste_length)

# 4. 命令调用者
class DocumentEditor:
    """文档编辑器 - 命令调用者"""
    
    def __init__(self):
        self.document = Document()
        self._command_history: List[Command] = []
        self._undo_history: List[Command] = []
    
    def execute_command(self, command: Command) -> None:
        """执行命令"""
        command.execute()
        self._command_history.append(command)
        self._undo_history.clear()  # 清除重做历史
    
    def undo(self) -> None:
        """撤销命令"""
        if not self._command_history:
            return
        
        command = self._command_history.pop()
        command.undo()
        self._undo_history.append(command)
    
    def redo(self) -> None:
        """重做命令"""
        if not self._undo_history:
            return
        
        command = self._undo_history.pop()
        command.execute()
        self._command_history.append(command)
    
    def get_content(self) -> str:
        """获取文档内容"""
        return self.document.get_content()

# 5. 宏命令
class MacroCommand(Command):
    """宏命令 - 组合多个命令"""
    
    def __init__(self, commands: List[Command]):
        self.commands = commands
    
    def execute(self) -> None:
        for command in self.commands:
            command.execute()
    
    def undo(self) -> None:
        for command in reversed(self.commands):
            command.undo()

# 6. 使用示例
def command_demo():
    print("命令模式示例：")
    
    editor = DocumentEditor()
    
    # 基本编辑操作
    print("\n1. 基本编辑操作:")
    # 插入文本
    editor.execute_command(InsertCommand(editor.document, "Hello, "))
    editor.execute_command(InsertCommand(editor.document, "World!"))
    print(f"当前内容: {editor.get_content()}")
    
    # 撤销操作
    print("\n2. 撤销操作:")
    editor.undo()
    print(f"撤销后内容: {editor.get_content()}")
    
    # 重做操作
    print("\n3. 重做操作:")
    editor.redo()
    print(f"重做后内容: {editor.get_content()}")
    
    # 复杂编辑操作
    print("\n4. 复杂编辑操作:")
    # 删除部分文本
    editor.execute_command(DeleteCommand(editor.document, 0, 6))
    print(f"删除后内容: {editor.get_content()}")
    
    # 复制粘贴操作
    editor.execute_command(CopyPasteCommand(editor.document, 0, 5, 5))
    print(f"复制粘贴后内容: {editor.get_content()}")
    
    # 宏命令示例
    print("\n5. 宏命令示例:")
    macro = MacroCommand([
        DeleteCommand(editor.document, 0, len(editor.get_content())),
        InsertCommand(editor.document, "New "),
        InsertCommand(editor.document, "Content"),
        CopyPasteCommand(editor.document, 0, 3, 8)
    ])
    
    editor.execute_command(macro)
    print(f"执行宏命令后内容: {editor.get_content()}")
    
    # 撤销宏命令
    print("\n6. 撤销宏命令:")
    editor.undo()
    print(f"撤销宏命令后内容: {editor.get_content()}")

if __name__ == "__main__":
    command_demo() 