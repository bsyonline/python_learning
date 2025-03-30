# Python IO操作 - 二进制IO示例

import struct
import array
import binascii

# 1. 基本二进制文件操作
def basic_binary_operations():
    print("基本二进制操作示例：")
    
    # 写入二进制数据
    with open('binary.dat', 'wb') as f:
        # 写入一些字节
        f.write(bytes([1, 2, 3, 4, 5]))
        # 写入字符串的字节表示
        f.write('Hello, Binary'.encode('utf-8'))
    print("二进制数据写入完成")
    
    # 读取二进制数据
    with open('binary.dat', 'rb') as f:
        # 读取前5个字节
        first_five = f.read(5)
        print(f"前5个字节: {list(first_five)}")
        # 读取剩余数据并解码
        rest = f.read()
        print(f"剩余数据解码: {rest.decode('utf-8')}")

# 2. 结构化二进制数据
def structured_binary_data():
    print("\n结构化二进制数据示例：")
    
    # 定义结构
    struct_format = 'i2f3s'  # int, 2 floats, 3 chars
    
    # 打包数据
    data = struct.pack(struct_format, 
                      123,           # int
                      2.718, 3.142,  # 2 floats
                      b'ABC')        # 3 chars
    print(f"打包后的数据: {binascii.hexlify(data)}")
    
    # 解包数据
    unpacked = struct.unpack(struct_format, data)
    print(f"解包后的数据: {unpacked}")
    
    # 使用array模块
    arr = array.array('i', [1, 2, 3, 4, 5])
    with open('array.bin', 'wb') as f:
        arr.tofile(f)
    print("\n数组已写入文件")
    
    # 从文件读取array
    arr2 = array.array('i')
    with open('array.bin', 'rb') as f:
        arr2.fromfile(f, 5)
    print(f"从文件读取的数组: {arr2.tolist()}")

# 3. 二进制数据处理
def binary_data_processing():
    print("\n二进制数据处理示例：")
    
    # 字节序操作
    value = 12345
    # 转换为大端字节序
    big_endian = value.to_bytes(4, byteorder='big')
    # 转换为小端字节序
    little_endian = value.to_bytes(4, byteorder='little')
    
    print(f"原始值: {value}")
    print(f"大端字节序: {list(big_endian)}")
    print(f"小端字节序: {list(little_endian)}")
    
    # 位操作
    flags = 0b00001111
    mask = 0b00000100
    
    result = flags & mask
    print(f"\n位操作:")
    print(f"flags: {bin(flags)}")
    print(f"mask:  {bin(mask)}")
    print(f"result:{bin(result)}")

# 4. 内存视图
def memory_views():
    print("\n内存视图示例：")
    
    # 创建字节数组
    data = bytearray(b'Hello, MemoryView!')
    
    # 创建内存视图
    view = memoryview(data)
    
    # 使用内存视图而不复制数据
    print(f"原始数据: {bytes(view)}")
    print(f"反转视图: {bytes(view[::-1])}")
    
    # 修改通过视图
    if view.readonly:
        print("视图是只读的")
    else:
        view[0] = ord('h')
        print(f"修改后的数据: {bytes(view)}")

# 5. 二进制文件格式处理
def binary_file_formats():
    print("\n二进制文件格式示例：")
    
    # 创建简单的二进制文件格式
    def write_record(f, id, name, age):
        name_bytes = name.encode('utf-8')
        # 写入固定长度记录
        f.write(struct.pack('i20si', 
                          id,
                          name_bytes + b'\x00' * (20 - len(name_bytes)),
                          age))
    
    # 写入记录
    with open('records.bin', 'wb') as f:
        write_record(f, 1, "Alice", 25)
        write_record(f, 2, "Bob", 30)
        write_record(f, 3, "Charlie", 35)
    
    # 读取记录
    def read_record(f):
        data = f.read(28)  # 4 + 20 + 4 bytes
        if not data:
            return None
        id, name, age = struct.unpack('i20si', data)
        return (id, name.decode('utf-8').rstrip('\x00'), age)
    
    print("\n读取记录:")
    with open('records.bin', 'rb') as f:
        while True:
            record = read_record(f)
            if record is None:
                break
            print(f"ID: {record[0]}, 名字: {record[1]}, 年龄: {record[2]}")

if __name__ == "__main__":
    basic_binary_operations()
    structured_binary_data()
    binary_data_processing()
    memory_views()
    binary_file_formats() 