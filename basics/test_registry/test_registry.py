import foo.operator as op
from foo.operator import Operator
from registry import Registry

print(op.multiply(1, 2))

def multiply(a, b):
    print("matrix multiply")
    return a * b

Registry.register("foo.operator.multiply", multiply)
print(op.multiply(1, 2))

def add(a, b):
    print("new add")
    return a + b

op = Operator()
print(op.add(1, 2))


Registry.register("foo.operator.Operator.add", add)
print(op.add(1, 2))