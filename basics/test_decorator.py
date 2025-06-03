
def decorator(func):
    """简单的装饰器"""
    def wrapper(*args, **kwargs):
        print("函数调用前")
        result = func(*args, **kwargs)
        print("函数调用后")
        return result
    return wrapper


if __name__ == "__main__":
    # 测试装饰器
    @decorator
    def say_hello(name):
        """被装饰的函数"""
        return f"你好，{name}！"
    print(say_hello("王五"))  # 输出装饰器的前后调用信息和函数结果
