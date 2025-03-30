# Python数据结构 - 字典操作

# 1. 字典的创建和基本操作
def basic_dict_operations():
    print("基本字典操作示例：")
    
    # 创建字典
    student = {
        "name": "张三",
        "age": 20,
        "grade": "A"
    }
    print(f"原始字典: {student}")
    
    # 添加/修改元素
    student["score"] = 95
    print(f"添加元素后: {student}")
    
    # 获取元素
    print(f"获取姓名: {student.get('name')}")
    print(f"获取不存在的键(带默认值): {student.get('address', '未知')}")
    
    # 删除元素
    removed = student.pop("grade")
    print(f"删除后的字典: {student}")
    print(f"被删除的值: {removed}")

# 2. 字典的遍历
def dict_iteration():
    print("\n字典遍历示例：")
    info = {
        "name": "李四",
        "age": 25,
        "city": "北京"
    }
    
    # 遍历键
    print("键遍历:")
    for key in info.keys():
        print(key)
    
    # 遍历值
    print("\n值遍历:")
    for value in info.values():
        print(value)
    
    # 遍历键值对
    print("\n键值对遍历:")
    for key, value in info.items():
        print(f"{key}: {value}")

# 3. 字典推导式
def dict_comprehension():
    print("\n字典推导式示例：")
    
    # 创建平方数字典
    squares = {x: x**2 for x in range(5)}
    print(f"平方数字典: {squares}")
    
    # 条件字典推导式
    even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
    print(f"偶数平方字典: {even_squares}")

# 4. 嵌套字典
def nested_dict():
    print("\n嵌套字典示例：")
    
    # 创建嵌套字典
    school = {
        "class_1": {
            "张三": {"age": 18, "score": 90},
            "李四": {"age": 19, "score": 85}
        },
        "class_2": {
            "王五": {"age": 18, "score": 95},
            "赵六": {"age": 19, "score": 88}
        }
    }
    
    # 访问嵌套值
    print(f"张三的成绩: {school['class_1']['张三']['score']}")
    
    # 修改嵌套值
    school['class_2']['王五']['score'] = 98
    print(f"修改后王五的成绩: {school['class_2']['王五']['score']}")

if __name__ == "__main__":
    basic_dict_operations()
    dict_iteration()
    dict_comprehension()
    nested_dict() 