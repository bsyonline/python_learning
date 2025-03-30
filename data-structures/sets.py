# Python数据结构 - 集合操作

# 1. 集合的创建和基本操作
def basic_set_operations():
    print("基本集合操作示例：")
    
    # 创建集合
    fruits = {"苹果", "香蕉", "橙子"}
    print(f"原始集合: {fruits}")
    
    # 添加元素
    fruits.add("葡萄")
    print(f"添加元素后: {fruits}")
    
    # 删除元素
    fruits.remove("香蕉")
    print(f"删除元素后: {fruits}")
    
    # 安全删除
    fruits.discard("不存在的水果")  # 不会引发错误
    print(f"安全删除后: {fruits}")

# 2. 集合运算
def set_operations():
    print("\n集合运算示例：")
    
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}
    
    # 并集
    union = set1 | set2
    print(f"并集: {union}")
    
    # 交集
    intersection = set1 & set2
    print(f"交集: {intersection}")
    
    # 差集
    difference = set1 - set2
    print(f"差集: {difference}")
    
    # 对称差集
    symmetric_difference = set1 ^ set2
    print(f"对称差集: {symmetric_difference}")

# 3. 集合推导式
def set_comprehension():
    print("\n集合推导式示例：")
    
    # 创建平方数集合
    squares = {x**2 for x in range(5)}
    print(f"平方数集合: {squares}")
    
    # 带条件的集合推导式
    even_squares = {x**2 for x in range(10) if x % 2 == 0}
    print(f"偶数的平方集合: {even_squares}")

# 4. 集合的应用
def set_applications():
    print("\n集合应用示例：")
    
    # 去重
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    unique_numbers = list(set(numbers))
    print(f"原列表: {numbers}")
    print(f"去重后: {unique_numbers}")
    
    # 成员检测
    fruits = {"苹果", "香蕉", "橙子"}
    print(f"'苹果' 在集合中: {'苹果' in fruits}")
    print(f"'葡萄' 在集合中: {'葡萄' in fruits}")

if __name__ == "__main__":
    basic_set_operations()
    set_operations()
    set_comprehension()
    set_applications() 