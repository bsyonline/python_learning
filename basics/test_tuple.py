# 定义
tuple1 = ('a', True, 1, 1.1, [1, 2, 3])
print(tuple1)  # ('a', True, 1, 1.1, [1, 2, 3])
print(type(tuple1))  # <class 'tuple'>
a, b, *c = tuple1
print(a)  # a
print(b)  # True
print(c)  # [1, 1.1, [1, 2, 3]]
a, b, _, _, _ = tuple1
print(a)  # a
print(b)  # True
t = ()
t1 = (1, )

# 获取元素
print(tuple1[0])  # a

# 元组的元素不可修改
# tuple[0] = 'b' # TypeError: 'tuple' object does not support item assignment
# 元组的元素可以是列表，列表的元素是可以修改的
tuple1[4][0] = 2
print(tuple1)  # ('a', True, 1, 1.1, [2, 2, 3])

# 遍历
for t in tuple1:
    print(t)  # a True 1 1.1 [2, 2, 3]

# 元组的方法
# 获取元素的索引
print(tuple1.index(1))  # 2
# 获取元素的个数
print(tuple1.count(1))  # 1
# 获取元组的长度
print(len(tuple1))  # 5

# 元组的合并
tuple2 = (1, 2, 3)
tuple3 = (4, 5, 6)
tuple4 = tuple2 + tuple3
print(tuple3)  # (1, 2, 3, 4, 5, 6)
tuple3 = (*tuple2, *tuple3)
print(tuple4)  # (1, 2, 3, 4, 5, 6)

# 元组的复制
tuple5 = tuple4[:]
print(tuple4)  # (1, 2, 3, 4, 5, 6)

# 元组的排序
tuple6 = (3, 2, 1)
tuple7 = sorted(tuple6)
print(tuple7)  # [1, 2, 3]

# 元组的反转
tuple8 = (1, 2, 3)
tuple9 = tuple8[::-1]
print(tuple9)  # (3, 2, 1)

# 元组的清空
tuple10 = (1, 2, 3)
tuple10 = ()
print(tuple10)  # ()
# 元组的删除
tuple11 = (1, 2, 3)
del tuple11

# 元组的切片
tuple12 = (1, 2, 3, 4, 5)
print(tuple12[1:3])  # (2, 3)
print(tuple12[1:])  # (2, 3, 4, 5)
print(tuple12[:3])  # (1, 2, 3)
print(tuple12[-1])  # 5
print(tuple12[-3:-1])  # (3, 4)

# 元组的拷贝
tuple13 = (1, 2, 3)
tuple14 = tuple13
print(tuple14)  # (1, 2, 3)

# 元组的比较
tuple15 = (1, 2, 3)
tuple16 = (1, 2, 3)
print(tuple15 == tuple16)  # True
print(tuple15 != tuple16)  # False
print(tuple15 > tuple16)  # False

# 元组的解包
a, b, c = (1, 2, 3)
print(a)  # 1

# 将元组转换成列表
li = list(tuple1)
print(li)  # ['a', True, 1, 1.1, [2, 2, 3]]

# 将列表转换成元组
tuple17 = tuple(li)
print(tuple17)  # ('a', True, 1, 1.1, [2, 2, 3])

# *args只能放在最后一个参数位置
def add(*args):
    print(args[0])
    print(args[1])
    print(args[2])
    total = 0
    for arg in args:
        print(arg)
        total += arg
    return total

add(1, 2, 3)

def point(x, y):
    return f'({x},{y})'

a = (0, 0)
origin = point(*a) # 运算符*会自动解包
print(origin)