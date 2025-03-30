# Python设计模式 - 享元模式
# 运用共享技术有效地支持大量细粒度的对象，避免重复创建相同的对象

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import json

# 1. 享元接口
class Character(ABC):
    """字符接口"""
    
    @abstractmethod
    def display(self, font_size: int, font_color: str) -> None:
        """显示字符"""
        pass

# 2. 具体享元类
class ConcreteCharacter(Character):
    """具体字符类"""
    
    def __init__(self, char: str):
        self._char = char
        # 内部状态 - 字符本身
        print(f"创建字符: {char}")
    
    def display(self, font_size: int, font_color: str) -> None:
        # 外部状态 - 字体大小和颜色
        print(f"显示: {self._char} [大小={font_size}, 颜色={font_color}]")

# 3. 享元工厂
class CharacterFactory:
    """字符工厂"""
    
    def __init__(self):
        self._characters: Dict[str, Character] = {}
    
    def get_character(self, char: str) -> Character:
        """获取字符对象，如果不存在则创建"""
        if char not in self._characters:
            self._characters[char] = ConcreteCharacter(char)
        return self._characters[char]
    
    def get_count(self) -> int:
        """获取已创建的字符数量"""
        return len(self._characters)

# 4. 非享元类
class TextContext:
    """文本上下文"""
    
    def __init__(self, font_size: int, font_color: str):
        self.font_size = font_size
        self.font_color = font_color

# 5. 文本编辑器
class TextEditor:
    """文本编辑器"""
    
    def __init__(self):
        self._factory = CharacterFactory()
        self._contexts: List[Tuple[Character, TextContext]] = []
    
    def append_text(self, text: str, context: TextContext) -> None:
        """添加文本"""
        for char in text:
            character = self._factory.get_character(char)
            self._contexts.append((character, context))
    
    def display(self) -> None:
        """显示所有文本"""
        for character, context in self._contexts:
            character.display(context.font_size, context.font_color)
    
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        return {
            "unique_chars": self._factory.get_count(),
            "total_chars": len(self._contexts)
        }

# 6. 高级应用 - 格式化文档
class DocumentStyle:
    """文档样式"""
    
    def __init__(self, name: str, font_size: int, font_color: str):
        self.name = name
        self.font_size = font_size
        self.font_color = font_color

class Document:
    """文档类"""
    
    def __init__(self):
        self._editor = TextEditor()
        self._styles: Dict[str, DocumentStyle] = {}
    
    def add_style(self, style: DocumentStyle) -> None:
        """添加样式"""
        self._styles[style.name] = style
    
    def append_text(self, text: str, style_name: str) -> None:
        """添加带样式的文本"""
        if style_name not in self._styles:
            raise ValueError(f"未知的样式: {style_name}")
        
        style = self._styles[style_name]
        context = TextContext(style.font_size, style.font_color)
        self._editor.append_text(text, context)
    
    def display(self) -> None:
        """显示文档"""
        self._editor.display()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self._editor.get_memory_usage()

# 7. 使用示例
def flyweight_demo():
    print("享元模式示例：")
    
    # 基本示例
    print("\n1. 基本文本编辑器示例:")
    editor = TextEditor()
    
    # 使用不同样式添加文本
    context1 = TextContext(12, "black")
    context2 = TextContext(14, "red")
    
    editor.append_text("Hello", context1)
    editor.append_text("World", context2)
    
    print("\n显示文本:")
    editor.display()
    
    memory_usage = editor.get_memory_usage()
    print(f"\n内存使用情况: {json.dumps(memory_usage, indent=2)}")
    
    # 高级文档示例
    print("\n2. 格式化文档示例:")
    document = Document()
    
    # 添加样式
    document.add_style(DocumentStyle("normal", 12, "black"))
    document.add_style(DocumentStyle("title", 16, "blue"))
    document.add_style(DocumentStyle("highlight", 14, "red"))
    
    # 添加带样式的文本
    document.append_text("这是标题", "title")
    document.append_text("这是正常文本", "normal")
    document.append_text("这是高亮文本", "highlight")
    
    print("\n显示文档:")
    document.display()
    
    stats = document.get_stats()
    print(f"\n文档统计: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    flyweight_demo() 