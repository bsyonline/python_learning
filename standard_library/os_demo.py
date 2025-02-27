# Python标准库 - 操作系统接口

import os
import shutil
from datetime import datetime

# 1. 路径操作
def path_operations():
    print("路径操作示例：")
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 路径拼接
    new_path = os.path.join(current_dir, "test_folder", "test.txt")
    print(f"拼接路径: {new_path}")
    
    # 获取路径信息
    print(f"目录名: {os.path.dirname(new_path)}")
    print(f"文件名: {os.path.basename(new_path)}")
    print(f"文件名和扩展名: {os.path.splitext(os.path.basename(new_path))}")

# 2. 文件和目录操作
def file_directory_operations():
    print("\n文件和目录操作示例：")
    
    # 创建目录
    os.makedirs("test_folder", exist_ok=True)
    print("创建目录: test_folder")
    
    # 创建文件
    with open("test_folder/test.txt", "w") as f:
        f.write("Hello, Python!")
    print("创建文件: test.txt")
    
    # 列出目录内容
    print("\n目录内容:")
    for item in os.listdir("test_folder"):
        print(item)
    
    # 获取文件信息
    file_stat = os.stat("test_folder/test.txt")
    print(f"\n文件大小: {file_stat.st_size} bytes")
    print(f"最后修改时间: {datetime.fromtimestamp(file_stat.st_mtime)}")

# 3. 环境变量操作
def environment_variables():
    print("\n环境变量操作示例：")
    
    # 获取所有环境变量
    print("部分环境变量:")
    for key, value in list(os.environ.items())[:3]:
        print(f"{key}: {value}")
    
    # 获取特定环境变量
    path = os.environ.get("PATH")
    print(f"\nPATH环境变量的前100个字符: {path[:100]}...")

# 4. 文件操作高级功能
def advanced_file_operations():
    print("\n高级文件操作示例：")
    
    # 复制文件
    shutil.copy2("test_folder/test.txt", "test_folder/test_backup.txt")
    print("文件已复制")
    
    # 移动文件
    os.rename("test_folder/test_backup.txt", "test_folder/test_moved.txt")
    print("文件已移动")
    
    # 删除文件
    os.remove("test_folder/test_moved.txt")
    print("文件已删除")
    
    # 删除目录
    shutil.rmtree("test_folder")
    print("目录已删除")

if __name__ == "__main__":
    path_operations()
    file_directory_operations()
    environment_variables()
    advanced_file_operations() 