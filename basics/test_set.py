# 创建集合
set = {1, 2, 3, 3, 3, 4, 5}
print(set)  # {1, 2, 3, 4, 5}
print(type(set))  # <class 'set'>
# 集合的长度
print(len(set))  # 5
# 遍历集合
for s in set:
    print(s)  # 1 2 3 4 5

# 集合的方法
# 添加单个元素
set.add(6)
print(set)  # {1, 2, 3, 4, 5, 6}
# 添加多个元素
set.update([7, 8])
print(set)  # {1, 2, 3, 4, 5, 6, 7, 8}

# 删除元素，如果元素不存在于集合中，它会引发KeyError异常
set.remove(6)
print(set)  # {1, 2, 3, 4, 5}
# 删除元素，如果元素不存在于集合中，它什么都不会做
set.discard(5)
print(set)  # {1, 2, 3, 4}
# 删除元素，随机删除一个元素
set.pop()
print(set)  # {2, 3, 4}
# 清空集合
set.clear()
print(set)  # set()
# 删除集合
del set
# print(set) # NameError: name 'set' is not defined

# 集合的运算
set1 = {1, 2, 3}
set2 = {3, 4, 5}
# 集合的并集，去重
set3 = set1 | set2
print(set3)  # {1, 2, 3, 4, 5}
# 集合的交集，取相同的元素
set4 = set1 & set2
print(set4)  # {3}
# 集合的差集，取不同的元素
set5 = set1 - set2
set6 = set2 - set1
print(set5)  # {1, 2}
print(set6)  # {4, 5}
# 集合的对称差集
set7 = set1 ^ set2
print(set6)  # {1, 2, 4, 5}
# 集合的子集判断
print(set1.issubset(set3))  # True
print(set1 <= set3)  # True
# 集合的超集判断
print(set3.issuperset(set1))  # True
print(set3 >= set1)  # True
# 集合的相等
print(set1 == set2)  # False
# 集合的不相等
print(set1 != set2)  # True
# 集合的拷贝
set8 = set1.copy()
print(set8)  # {1, 2, 3}

# 集合的转换
list = [1, 2, 3]
set9 = set(list)
print(set9)  # {1, 2, 3}

# 集合的排序
set10 = sorted(set9)
print(set10)  # [1, 2, 3]
