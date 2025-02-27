# Python基础 - 控制流

# 1. if-elif-else 条件语句
def demonstrate_if_statements():
    age = 18
    print("条件语句示例：")
    if age < 18:
        print("未成年")
    elif age == 18:
        print("刚好成年")
    else:
        print("成年人")

# 2. for循环
def demonstrate_for_loops():
    print("\nfor循环示例：")
    # 遍历列表
    fruits = ["苹果", "香蕉", "橙子"]
    for fruit in fruits:
        print(f"水果: {fruit}")
    
    # 使用range()
    print("\n使用range():")
    for i in range(3):
        print(f"计数: {i}")

# 3. while循环
def demonstrate_while_loops():
    print("\nwhile循环示例：")
    count = 0
    while count < 3:
        print(f"当前计数: {count}")
        count += 1

# 4. break和continue语句
def demonstrate_break_continue():
    print("\nbreak和continue示例：")
    for i in range(5):
        if i == 2:
            continue  # 跳过2
        if i == 4:
            break    # 到4时退出循环
        print(f"数字: {i}")

# 执行所有示例
if __name__ == "__main__":
    demonstrate_if_statements()
    demonstrate_for_loops()
    demonstrate_while_loops()
    demonstrate_break_continue() 