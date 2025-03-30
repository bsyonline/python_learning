# Python设计模式 - 组合模式
# 将对象组合成树形结构以表示"部分-整体"的层次结构，使得用户对单个对象和组合对象的使用具有一致性

from abc import ABC, abstractmethod
from typing import List, Optional

# 1. 组件抽象基类
class FileSystemComponent(ABC):
    """文件系统组件抽象类"""
    
    def __init__(self, name: str):
        self.name = name
        self._parent: Optional['Directory'] = None
    
    @property
    def parent(self) -> Optional['Directory']:
        return self._parent
    
    @parent.setter
    def parent(self, parent: 'Directory') -> None:
        self._parent = parent
    
    @abstractmethod
    def display(self, level: int = 0) -> None:
        """显示组件"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """获取大小"""
        pass
    
    def get_path(self) -> str:
        """获取路径"""
        if self.parent:
            return f"{self.parent.get_path()}/{self.name}"
        return self.name

# 2. 叶子组件
class File(FileSystemComponent):
    """文件类"""
    
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size = size
    
    def display(self, level: int = 0) -> None:
        indent = "  " * level
        print(f"{indent}文件: {self.name} ({self.size} bytes)")
    
    def get_size(self) -> int:
        return self.size

# 3. 组合组件
class Directory(FileSystemComponent):
    """目录类"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._children: List[FileSystemComponent] = []
    
    def add(self, component: FileSystemComponent) -> None:
        """添加组件"""
        self._children.append(component)
        component.parent = self
    
    def remove(self, component: FileSystemComponent) -> None:
        """移除组件"""
        if component in self._children:
            self._children.remove(component)
            component.parent = None
    
    def display(self, level: int = 0) -> None:
        indent = "  " * level
        print(f"{indent}目录: {self.name}/")
        for child in self._children:
            child.display(level + 1)
    
    def get_size(self) -> int:
        return sum(child.get_size() for child in self._children)

# 4. 文件系统操作类
class FileSystem:
    """文件系统类"""
    
    def __init__(self):
        self.root = Directory("")
    
    def create_directory(self, path: str) -> Directory:
        """创建目录"""
        parts = path.strip("/").split("/")
        current = self.root
        
        for part in parts:
            # 检查是否已存在该目录
            found = False
            for child in current._children:
                if isinstance(child, Directory) and child.name == part:
                    current = child
                    found = True
                    break
            
            if not found:
                new_dir = Directory(part)
                current.add(new_dir)
                current = new_dir
        
        return current
    
    def create_file(self, path: str, size: int) -> File:
        """创建文件"""
        dirname, filename = path.rsplit("/", 1) if "/" in path else ("", path)
        directory = self.create_directory(dirname) if dirname else self.root
        
        file = File(filename, size)
        directory.add(file)
        return file
    
    def display(self) -> None:
        """显示文件系统"""
        print("文件系统结构:")
        self.root.display()
    
    def get_total_size(self) -> int:
        """获取总大小"""
        return self.root.get_size()

# 5. 使用示例
def composite_demo():
    print("组合模式示例：")
    
    # 创建文件系统
    fs = FileSystem()
    
    # 创建目录结构
    print("\n1. 创建目录结构:")
    fs.create_directory("home/user1")
    fs.create_directory("home/user2")
    fs.create_directory("usr/local/bin")
    
    # 创建文件
    print("\n2. 创建文件:")
    fs.create_file("home/user1/document.txt", 1000)
    fs.create_file("home/user1/image.jpg", 2000)
    fs.create_file("home/user2/video.mp4", 5000)
    fs.create_file("usr/local/bin/program.exe", 3000)
    
    # 显示文件系统
    print("\n3. 显示整个文件系统:")
    fs.display()
    
    # 显示大小信息
    print(f"\n4. 文件系统总大小: {fs.get_total_size()} bytes")
    
    # 显示路径
    print("\n5. 文件路径示例:")
    doc = fs.create_file("home/user1/notes.txt", 500)
    print(f"文件完整路径: {doc.get_path()}")
    
    # 目录操作示例
    print("\n6. 目录操作示例:")
    projects_dir = fs.create_directory("home/user1/projects")
    src_file = File("main.py", 800)
    projects_dir.add(src_file)
    
    print("更新后的文件系统:")
    fs.display()

if __name__ == "__main__":
    composite_demo() 