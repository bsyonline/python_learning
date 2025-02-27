# 简单计算器实现

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, x, y):
        """加法运算"""
        result = x + y
        self.history.append(f"{x} + {y} = {result}")
        return result
    
    def subtract(self, x, y):
        """减法运算"""
        result = x - y
        self.history.append(f"{x} - {y} = {result}")
        return result
    
    def multiply(self, x, y):
        """乘法运算"""
        result = x * y
        self.history.append(f"{x} × {y} = {result}")
        return result
    
    def divide(self, x, y):
        """除法运算"""
        if y == 0:
            raise ValueError("除数不能为零")
        result = x / y
        self.history.append(f"{x} ÷ {y} = {result}")
        return result
    
    def get_history(self):
        """获取计算历史"""
        return self.history

def main():
    calc = Calculator()
    
    while True:
        print("\n简单计算器")
        print("1. 加法")
        print("2. 减法")
        print("3. 乘法")
        print("4. 除法")
        print("5. 显示历史记录")
        print("6. 退出")
        
        choice = input("请选择操作 (1-6): ")
        
        if choice == '6':
            print("感谢使用！")
            break
        
        if choice == '5':
            history = calc.get_history()
            if history:
                print("\n计算历史:")
                for item in history:
                    print(item)
            else:
                print("暂无计算历史")
            continue
        
        try:
            x = float(input("请输入第一个数字: "))
            y = float(input("请输入第二个数字: "))
            
            if choice == '1':
                result = calc.add(x, y)
                print(f"结果: {result}")
            elif choice == '2':
                result = calc.subtract(x, y)
                print(f"结果: {result}")
            elif choice == '3':
                result = calc.multiply(x, y)
                print(f"结果: {result}")
            elif choice == '4':
                try:
                    result = calc.divide(x, y)
                    print(f"结果: {result}")
                except ValueError as e:
                    print(f"错误: {e}")
            else:
                print("无效的选择")
        
        except ValueError:
            print("请输入有效的数字")

if __name__ == "__main__":
    main() 