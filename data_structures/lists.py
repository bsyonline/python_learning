# Python数据结构 - 列表操作

# 1. 列表的创建和基本操作
def basic_list_operations():
    print("基本列表操作示例：")
    
    # 创建列表
    fruits = ["苹果", "香蕉", "橙子"]
    print(f"原始列表: {fruits}")
    
    # 添加元素
    fruits.append("葡萄")
    print(f"append后: {fruits}")
    
    # 插入元素
    fruits.insert(1, "梨")
    print(f"insert后: {fruits}")
    
    # 删除元素
    removed = fruits.pop()
    print(f"pop后: {fruits}")
    print(f"被删除的元素: {removed}")

# 2. 列表切片
def list_slicing():
    print("\n列表切片示例：")
    numbers = [0, 1, 2, 3, 4, 5]
    print(f"完整列表: {numbers}")
    print(f"前三个元素: {numbers[:3]}")
    print(f"第2个到第4个: {numbers[1:4]}")
    print(f"间隔取值: {numbers[::2]}")

# 3. 列表推导式
def list_comprehension():
    print("\n列表推导式示例：")
    
    # 生成平方数
    squares = [x**2 for x in range(5)]
    print(f"平方数列表: {squares}")
    
    # 带条件的列表推导式
    even_squares = [x**2 for x in range(10) if x % 2 == 0]
    print(f"偶数的平方: {even_squares}")

# 4. 列表的排序和反转
def list_sorting():
    print("\n列表排序示例：")
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    
    # 排序
    sorted_numbers = sorted(numbers)
    print(f"排序后: {sorted_numbers}")
    
    # 反转
    numbers.reverse()
    print(f"反转后: {numbers}")
    
    # 自定义排序
    fruits = ["apple", "banana", "Orange", "grape"]
    sorted_fruits = sorted(fruits, key=str.lower)
    print(f"忽略大小写排序: {sorted_fruits}")

if __name__ == "__main__":
    basic_list_operations()
    list_slicing()
    list_comprehension()
    list_sorting() 