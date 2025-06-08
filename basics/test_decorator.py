
def decorator(func):
    """简单的装饰器"""
    def wrapper(*args, **kwargs):
        print("函数调用前")
        result = func(*args, **kwargs)
        print("函数调用后")
        return result
    return wrapper


class class_decorator(object):
    """简单的类装饰器"""
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("类装饰器调用前")
        result = self.func(*args, **kwargs)
        print("类装饰器调用后")
        return result



if __name__ == "__main__":
    # 显示调用装饰器函数
    def hello():
        print("hello world!")

    hello_decorator = decorator(hello)
    hello_decorator()


    # 测试装饰器
    @decorator
    def hello():
        """被装饰的函数"""
        print("hello world!")

    hello()  # 输出装饰器的前后调用信息和函数结果
    print(hello.__name__) # warning: 原函数名被wrapper覆盖了，所以这里输出的是wrapper而不是hello

    @class_decorator
    def hello():
        """被类装饰器装饰的函数"""
        print("hello world!")

    hello()  # 输出类装饰器的前后调用信息和函数结果
