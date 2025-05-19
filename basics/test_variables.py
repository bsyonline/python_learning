# Python基础 - 变量和数据类型

# 1. 数值类型
integer_number = 42  # 整数
float_number = 3.14  # 浮点数
complex_number = 1 + 2j  # 复数

print("数值类型示例：")
print(f"整数: {integer_number}")
print(f"浮点数: {float_number}")
print(f"复数: {complex_number}")

# 2. 字符串
string_single = 'Hello'  # 单引号字符串
string_double = "World"  # 双引号字符串
string_multi = '''这是一个
多行字符串'''  # 多行字符串

print("\n字符串示例：")
print(f"字符串拼接: {string_single} {string_double}")
print(f"多行字符串:\n{string_multi}")

# 3. 布尔类型
is_true = True
is_false = False

print("\n布尔类型示例：")
print(f"True值: {is_true}")
print(f"False值: {is_false}")

# 4. 类型转换
string_number = "123"
converted_number = int(string_number)

print("\n类型转换示例：")
print(f"字符串 '{string_number}' 转换为整数: {converted_number}")
print(f"类型: {type(converted_number)}")

# 5. 变量命名规范示例
user_name = "张三"      # 使用下划线分隔单词
firstName = "李"        # 驼峰命名法
COUNT_MAX = 100        # 常量通常使用大写 