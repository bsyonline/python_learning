# Python设计模式 - 解释器模式
# 给定一个语言，定义它的文法的一种表示，并定义一个解释器，这个解释器使用该表示来解释语言中的句子

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# 1. 抽象表达式
class Expression(ABC):
    """表达式抽象基类"""
    
    @abstractmethod
    def interpret(self, context: Dict[str, int]) -> int:
        """解释表达式"""
        pass

# 2. 终结符表达式
class NumberExpression(Expression):
    """数字表达式"""
    
    def __init__(self, number: int):
        self.number = number
    
    def interpret(self, context: Dict[str, int]) -> int:
        return self.number

class VariableExpression(Expression):
    """变量表达式"""
    
    def __init__(self, name: str):
        self.name = name
    
    def interpret(self, context: Dict[str, int]) -> int:
        if self.name not in context:
            raise ValueError(f"变量 {self.name} 未定义")
        return context[self.name]

# 3. 非终结符表达式
class AddExpression(Expression):
    """加法表达式"""
    
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Dict[str, int]) -> int:
        return self.left.interpret(context) + self.right.interpret(context)

class SubtractExpression(Expression):
    """减法表达式"""
    
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Dict[str, int]) -> int:
        return self.left.interpret(context) - self.right.interpret(context)

class MultiplyExpression(Expression):
    """乘法表达式"""
    
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Dict[str, int]) -> int:
        return self.left.interpret(context) * self.right.interpret(context)

# 4. 高级表达式
class AssignExpression(Expression):
    """赋值表达式"""
    
    def __init__(self, name: str, expression: Expression):
        self.name = name
        self.expression = expression
    
    def interpret(self, context: Dict[str, int]) -> int:
        value = self.expression.interpret(context)
        context[self.name] = value
        return value

class ConditionExpression(Expression):
    """条件表达式"""
    
    def __init__(self, condition: Expression, if_true: Expression, if_false: Expression):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
    
    def interpret(self, context: Dict[str, int]) -> int:
        if self.condition.interpret(context) != 0:
            return self.if_true.interpret(context)
        return self.if_false.interpret(context)

# 5. 解释器上下文
class ExpressionParser:
    """表达式解析器"""
    
    def parse_number(self, token: str) -> Expression:
        """解析数字"""
        return NumberExpression(int(token))
    
    def parse_variable(self, token: str) -> Expression:
        """解析变量"""
        return VariableExpression(token)
    
    def parse_binary_operation(self, left: str, operator: str, right: str) -> Expression:
        """解析二元操作"""
        left_expr = self.parse_token(left)
        right_expr = self.parse_token(right)
        
        if operator == "+":
            return AddExpression(left_expr, right_expr)
        elif operator == "-":
            return SubtractExpression(left_expr, right_expr)
        elif operator == "*":
            return MultiplyExpression(left_expr, right_expr)
        else:
            raise ValueError(f"未知的操作符: {operator}")
    
    def parse_token(self, token: str) -> Expression:
        """解析单个标记"""
        try:
            return self.parse_number(token)
        except ValueError:
            return self.parse_variable(token)
    
    def parse_assignment(self, variable: str, expression: str) -> Expression:
        """解析赋值语句"""
        return AssignExpression(variable, self.parse_token(expression))
    
    def parse_condition(self, condition: str, if_true: str, if_false: str) -> Expression:
        """解析条件语句"""
        return ConditionExpression(
            self.parse_token(condition),
            self.parse_token(if_true),
            self.parse_token(if_false)
        )

# 6. 使用示例
def interpreter_demo():
    print("解释器模式示例：")
    
    # 创建解析器和上下文
    parser = ExpressionParser()
    context: Dict[str, int] = {}
    
    # 基本算术表达式
    print("\n1. 基本算术表达式:")
    expr1 = parser.parse_binary_operation("5", "+", "3")
    print(f"5 + 3 = {expr1.interpret(context)}")
    
    expr2 = parser.parse_binary_operation("10", "*", "2")
    print(f"10 * 2 = {expr2.interpret(context)}")
    
    # 变量赋值和使用
    print("\n2. 变量操作:")
    assign1 = parser.parse_assignment("x", "42")
    print(f"x = 42 -> {assign1.interpret(context)}")
    
    expr3 = parser.parse_binary_operation("x", "-", "10")
    print(f"x - 10 = {expr3.interpret(context)}")
    
    # 复杂表达式
    print("\n3. 复杂表达式:")
    # (x + 5) * 2
    complex_expr = MultiplyExpression(
        AddExpression(
            VariableExpression("x"),
            NumberExpression(5)
        ),
        NumberExpression(2)
    )
    print(f"(x + 5) * 2 = {complex_expr.interpret(context)}")
    
    # 条件表达式
    print("\n4. 条件表达式:")
    # if x > 40 then 100 else 0
    condition = ConditionExpression(
        SubtractExpression(VariableExpression("x"), NumberExpression(40)),
        NumberExpression(100),
        NumberExpression(0)
    )
    print(f"if x > 40 then 100 else 0 = {condition.interpret(context)}")
    
    # 显示最终上下文
    print("\n5. 最终变量状态:")
    for name, value in context.items():
        print(f"{name} = {value}")

if __name__ == "__main__":
    interpreter_demo() 