# Python基础 - 函数

# 1. 基本函数定义和调用
def greet(name):
    """简单的问候函数"""
    return f"你好，{name}！"

# 2. 带默认参数的函数
def calculate_power(base, exponent=2):
    """计算幂，默认是平方"""
    return base ** exponent

# 3. 带多个返回值的函数
def get_statistics(numbers):
    """返回一个列表的最小值、最大值和平均值"""
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

# 4. 可变参数函数
def sum_all(*args):
    """计算所有传入参数的和"""
    return sum(args)

# 5. 关键字参数函数
def create_profile(**kwargs):
    """创建用户档案"""
    profile = ""
    for key, value in kwargs.items():
        profile += f"{key}: {value}\n"
    return profile

# 6. lambda 函数
square = lambda x: x ** 2

# 7. 内置函数
import builtins
print(dir(builtins)) # 打印所有内置函数和变量

# 8. 闭包
def outer_function(x):
    """外部函数，返回一个闭包"""
    def inner_function(y):
        """内部函数，使用外部函数的变量"""
        return x + y
    return inner_function



# 示例使用
if __name__ == "__main__":
    # 测试基本函数
    print("基本函数示例：")
    print(greet("张三"))
    
    # 测试带默认参数的函数
    print("\n默认参数示例：")
    print(f"2的平方: {calculate_power(2)}")
    print(f"2的3次方: {calculate_power(2, 3)}")
    
    # 测试多返回值函数
    print("\n多返回值示例：")
    numbers = [1, 2, 3, 4, 5]
    min_val, max_val, avg = get_statistics(numbers)
    print(f"最小值: {min_val}, 最大值: {max_val}, 平均值: {avg}")
    
    # 测试可变参数函数
    print("\n可变参数示例：")
    print(f"求和结果: {sum_all(1, 2, 3, 4)}")
    
    # 测试关键字参数函数
    print("\n关键字参数示例：")
    profile = create_profile(name="李四", age=25, city="北京")
    print(profile)

    # 测试 lambda 函数
    print("\nlambda 函数示例：")
    print(f"4的平方: {square(4)}")

    # 测试闭包
    print("\n闭包示例：")
    closure = outer_function(10)
    print(f"闭包结果: {closure(5)}")  # 输出 15

