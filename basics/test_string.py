# 定义一个字符串
s = "机器学习"
print(s, type(s))  # 打印字符串和它的类型

# 字符串编码：将字符串转换为字节序列
b = s.encode("utf-8")  # 使用UTF-8编码将字符串转换为字节
print(b, type(b))  # 打印字节序列和它的类型

# 字节解码：将字节序列转换回字符串
s1 = b.decode("utf-8")  # 使用UTF-8解码将字节转换回字符串
print(s1, type(s1))  # 打印解码后的字符串和它的类型

# 字符串连接操作
s2 = s + s1  # 连接两个字符串
print(s2)  # 打印连接后的结果

# 字符串重复操作
s3 = s * 3  # 重复字符串三次
print(s3)  # 打印重复后的结果

# 字符串成员检测
print("机器" in s)  # 检查子字符串"机器"是否在s中，返回True
print("机器" not in s)  # 检查子字符串"机器"是否不在s中，返回False

# 字符串索引访问
print(s[0])  # 打印第一个字符
print(s[-1])  # 打印最后一个字符

s4 = "abcdefghijklmnopqrstuvwxyz"
# 字符串切片操作
print(s4[0:2])  # 从索引0到索引1（不包括2）的子字符串

# 带步长的切片
print(s4[0:9:2])  # 从索引0到索引8，步长为2的子字符串
print(s4[0:9:-2])  # 从索引0到索引8，步长为-2的子字符串（这里结果为空，因为负步长需要起始索引大于结束索引）

# 负索引和负步长的切片
print(s4[-1])  # 打印最后一个字符
print(s4[-1:-9:2])  # 从倒数第1个到倒数第9个，步长为2的子字符串（这里结果为空，因为正步长需要起始索引小于结束索引）
print(s4[-1:-9:-2])  # 从倒数第1个到倒数第9个，步长为-2的子字符串

print(s.find("机"))  # 查找子字符串"机"在s中的索引位置
print(s.find("机", 3))  # 查找子字符串"机"在s中的索引位置，从索引3开始查找
print(s.find("机", 0, 2))  # 查找子字符串"机"在s中的索引位置，范围是0到2

print(s.rfind("机"))  # 从右侧查找子字符串"机"在s中的索引位置

print(s.index("机"))  # 查找子字符串"机"在s中的索引位置，如果不存在则抛出异常   
print(s.rindex("机"))  # 从右侧查找子字符串"机"在s中的索引位置，如果不存在则抛出异常

print(s.count("机"))  # 统计子字符串"机"在s中出现的次数

print(s.startswith("机"))  # 检查字符串s是否以"机"开头
print(s.endswith("机"))  # 检查字符串s是否以"机"结尾
print(s.startswith("机", 0, 2))  # 检查字符串s在范围0到2内是否以"机"开头
print(s.endswith("机", 0, 2))  # 检查字符串s在范围0到2内是否以"机"结尾

s5 = "hello world"
print(s5.capitalize())  # 将字符串的首字母大写
print(s5.title())  # 将字符串的每个单词的首字母大写

print(s5.upper())  # 将字符串转换为大写
print(s5.lower())  # 将字符串转换为小写

s6 = "Hello World"
print(s6.swapcase())  # 将字符串中的大写字母转换为小写字母，小写字母转换为大写字母

s7 = "  hello world  "
print(s7.strip())  # 去除字符串两端的空格    
print(s7.lstrip())  # 去除字符串左端的空格
print(s7.rstrip())  # 去除字符串右端的空格

print(s.replace("机器", "深度"))  # 替换字符串中的"机"为"人"

print(s.split("学"))  # 将字符串按"学"分割成列表
print(s.split("学", 1))  # 将字符串按"学"分割成列表，分割次数为1

