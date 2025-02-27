# Python IO操作 - 内存字符串IO示例

from io import StringIO, BytesIO
import csv

# 1. 基本StringIO操作
def basic_string_io():
    print("基本StringIO操作示例：")
    
    # 创建StringIO对象
    output = StringIO()
    
    # 写入数据
    output.write('Hello, ')
    output.write('StringIO!\n')
    print('这行会被写入StringIO', file=output)
    
    # 获取全部内容
    content = output.getvalue()
    print("StringIO内容:")
    print(content)
    
    # 从字符串创建StringIO
    input = StringIO('Line 1\nLine 2\nLine 3\n')
    
    # 逐行读取
    print("\n逐行读取:")
    for line in input:
        print(f"读取: {line.strip()}")
    
    # 关闭
    output.close()
    input.close()

# 2. CSV处理
def csv_string_io():
    print("\nCSV StringIO操作示例：")
    
    # 创建CSV数据
    output = StringIO()
    writer = csv.writer(output)
    
    # 写入数据
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerows([
        ['Alice', '25', 'Beijing'],
        ['Bob', '30', 'Shanghai'],
        ['Charlie', '35', 'Guangzhou']
    ])
    
    # 获取CSV内容
    csv_content = output.getvalue()
    print("CSV内容:")
    print(csv_content)
    
    # 读取CSV数据
    input = StringIO(csv_content)
    reader = csv.reader(input)
    
    print("\n解析CSV:")
    for row in reader:
        print(f"行数据: {row}")
    
    output.close()
    input.close()

# 3. 内存中的文本处理
def text_processing():
    print("\n内存中的文本处理示例：")
    
    # 创建文本缓冲区
    buffer = StringIO()
    
    # 文本处理函数
    def process_text(text):
        # 转换为大写并添加行号
        lines = text.split('\n')
        for i, line in enumerate(lines, 1):
            buffer.write(f"{i}. {line.upper()}\n")
    
    # 处理一些文本
    sample_text = """first line
    second line
    third line"""
    
    process_text(sample_text)
    
    # 显示处理结果
    print("处理后的文本:")
    print(buffer.getvalue())
    
    buffer.close()

# 4. BytesIO操作
def bytes_io_operations():
    print("\nBytesIO操作示例：")
    
    # 创建BytesIO对象
    binary_buffer = BytesIO()
    
    # 写入二进制数据
    binary_buffer.write(b'Hello, ')
    binary_buffer.write('世界'.encode('utf-8'))
    
    # 获取所有数据
    binary_data = binary_buffer.getvalue()
    print(f"原始二进制数据: {binary_data}")
    print(f"解码后的文本: {binary_data.decode('utf-8')}")
    
    # 使用seek和tell
    binary_buffer.seek(0)  # 回到开始
    print(f"\n当前位置: {binary_buffer.tell()}")
    
    # 读取部分数据
    part = binary_buffer.read(5)
    print(f"读取5个字节: {part}")
    print(f"当前位置: {binary_buffer.tell()}")
    
    binary_buffer.close()

# 5. 混合使用示例
def mixed_io_example():
    print("\n混合IO使用示例：")
    
    # 创建文本和二进制缓冲区
    text_buffer = StringIO()
    binary_buffer = BytesIO()
    
    # 写入数据
    text_buffer.write("这是文本数据\n")
    binary_buffer.write(b'\x00\x01\x02\x03\x04')
    
    # 转换文本到二进制
    text_data = text_buffer.getvalue()
    binary_buffer.write(text_data.encode('utf-8'))
    
    # 显示结果
    print(f"文本缓冲区内容: {text_buffer.getvalue()}")
    print(f"二进制缓冲区内容: {binary_buffer.getvalue()}")
    
    text_buffer.close()
    binary_buffer.close()

if __name__ == "__main__":
    basic_string_io()
    csv_string_io()
    text_processing()
    bytes_io_operations()
    mixed_io_example() 