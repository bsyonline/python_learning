# Python IO操作 - 文件IO示例

import os
from pathlib import Path

# 1. 基本文件操作
def basic_file_operations():
    print("基本文件操作示例：")
    
    # 写入文件
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write('Hello, Python!\n')
        f.write('这是第二行\n')
        # 写入多行
        lines = ['第三行\n', '第四行\n', '第五行\n']
        f.writelines(lines)
    print("文件写入完成")
    
    # 读取文件
    with open('test.txt', 'r', encoding='utf-8') as f:
        # 读取整个文件
        content = f.read()
        print("\n读取整个文件:")
        print(content)
    
    # 逐行读取
    with open('test.txt', 'r', encoding='utf-8') as f:
        print("\n逐行读取:")
        for line in f:
            print(f"行内容: {line.strip()}")

# 2. 文件指针操作
def file_pointer_operations():
    print("\n文件指针操作示例：")
    
    with open('test.txt', 'r+', encoding='utf-8') as f:
        # 读取前两个字符
        print(f"前两个字符: {f.read(2)}")
        
        # 获取当前位置
        print(f"当前位置: {f.tell()}")
        
        # 移动到文件开头
        f.seek(0)
        print(f"回到开头后读取: {f.readline().strip()}")
        
        # 移动到第二行开头
        f.seek(0)
        f.readline()  # 跳过第一行
        print(f"第二行内容: {f.readline().strip()}")

# 3. 文件属性和状态
def file_attributes():
    print("\n文件属性示例：")
    
    file_path = 'test.txt'
    # 使用os模块
    print(f"文件大小: {os.path.getsize(file_path)} 字节")
    print(f"最后修改时间: {os.path.getmtime(file_path)}")
    print(f"是否为文件: {os.path.isfile(file_path)}")
    print(f"是否为目录: {os.path.isdir(file_path)}")
    
    # 使用pathlib
    path = Path(file_path)
    print(f"\nPath对象信息:")
    print(f"文件名: {path.name}")
    print(f"后缀: {path.suffix}")
    print(f"父目录: {path.parent}")
    print(f"绝对路径: {path.absolute()}")

# 4. 高级文件操作
def advanced_file_operations():
    print("\n高级文件操作示例：")
    
    # 创建临时文件
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w+', delete=False) as temp:
        temp.write('临时文件内容\n')
        temp_name = temp.name
        print(f"临时文件创建在: {temp_name}")
    
    # 文件过滤器
    def line_filter(filename, prefix):
        with open(filename, 'r', encoding='utf-8') as f:
            return [line for line in f if line.startswith(prefix)]
    
    # 使用过滤器
    filtered_lines = line_filter('test.txt', '第')
    print("\n以'第'开头的行:")
    for line in filtered_lines:
        print(line.strip())
    
    # 清理临时文件
    os.unlink(temp_name)

# 5. 上下文管理器
class FileManager:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

def context_manager_demo():
    print("\n上下文管理器示例：")
    
    # 使用自定义上下文管理器
    with FileManager('test.txt', 'r') as f:
        content = f.read()
        print(f"使用自定义管理器读取内容长度: {len(content)}")

if __name__ == "__main__":
    basic_file_operations()
    file_pointer_operations()
    file_attributes()
    advanced_file_operations()
    context_manager_demo() 