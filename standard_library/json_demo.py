# Python标准库 - JSON处理

import json

# 1. 基本JSON操作
def basic_json_operations():
    print("基本JSON操作示例：")
    
    # Python字典
    person = {
        "name": "张三",
        "age": 30,
        "city": "北京",
        "hobbies": ["读书", "游泳", "编程"]
    }
    
    # 转换为JSON字符串
    json_str = json.dumps(person, ensure_ascii=False)
    print(f"转换为JSON:\n{json_str}")
    
    # 解析JSON字符串
    parsed_data = json.loads(json_str)
    print(f"\n解析JSON:\n{parsed_data}")

# 2. 文件读写
def json_file_operations():
    print("\nJSON文件操作示例：")
    
    # 写入JSON文件
    data = {
        "students": [
            {"name": "李四", "age": 18, "score": 95},
            {"name": "王五", "age": 19, "score": 88}
        ]
    }
    
    with open("students.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("数据已写入students.json")
    
    # 读取JSON文件
    with open("students.json", "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    print(f"\n读取的数据:\n{loaded_data}")

# 3. 自定义对象的JSON序列化
class Student:
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

def custom_object_serialization():
    print("\n自定义对象序列化示例：")
    
    # 自定义编码器
    class StudentEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Student):
                return {
                    "name": obj.name,
                    "age": obj.age,
                    "score": obj.score
                }
            return super().default(obj)
    
    # 创建学生对象
    student = Student("赵六", 20, 92)
    
    # 序列化
    json_str = json.dumps(student, cls=StudentEncoder, ensure_ascii=False)
    print(f"序列化结果:\n{json_str}")

# 4. JSON格式化输出
def formatted_json():
    print("\nJSON格式化输出示例：")
    
    data = {
        "name": "张三",
        "info": {
            "age": 30,
            "contact": {
                "email": "zhangsan@example.com",
                "phone": "123456789"
            }
        },
        "scores": [95, 88, 92]
    }
    
    # 格式化输出
    formatted = json.dumps(data, ensure_ascii=False, indent=4)
    print(formatted)

if __name__ == "__main__":
    basic_json_operations()
    json_file_operations()
    custom_object_serialization()
    formatted_json() 