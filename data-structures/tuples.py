# Python数据结构 - 元组操作

# 1. 元组的创建和基本操作
def basic_tuple_operations():
    print("基本元组操作示例：")
    
    # 创建元组
    coordinates = (10, 20)
    print(f"坐标: {coordinates}")
    
    # 创建单元素元组
    single_tuple = (1,)  # 注意逗号
    print(f"单元素元组: {single_tuple}")
    
    # 元组解包
    x, y = coordinates
    print(f"解包后 - x: {x}, y: {y}")
    
    # 尝试修改元组（会引发错误）
    try:
        coordinates[0] = 30
    except TypeError as e:
        print(f"修改元组错误: {e}")

# 2. 元组的使用场景
def tuple_usage():
    print("\n元组使用场景示例：")
    
    # 函数返回多个值
    def get_coordinates():
        return (100, 200)
    
    pos_x, pos_y = get_coordinates()
    print(f"获取坐标 - x: {pos_x}, y: {pos_y}")
    
    # 作为字典键
    point_values = {(0, 0): "原点", (1, 1): "对角线点"}
    print(f"点(0, 0)的值: {point_values[(0, 0)]}")

# 3. 元组的方法
def tuple_methods():
    print("\n元组方法示例：")
    
    numbers = (1, 2, 2, 3, 4, 2)
    
    # 计数
    print(f"2出现的次数: {numbers.count(2)}")
    
    # 查找索引
    print(f"第一个2的索引: {numbers.index(2)}")
    
    # 切片
    print(f"前三个元素: {numbers[:3]}")
    print(f"后两个元素: {numbers[-2:]}")

# 4. 元组与列表的转换
def tuple_list_conversion():
    print("\n元组与列表转换示例：")
    
    # 列表转元组
    list_numbers = [1, 2, 3, 4, 5]
    tuple_numbers = tuple(list_numbers)
    print(f"列表转元组: {tuple_numbers}")
    
    # 元组转列表
    tuple_fruits = ("苹果", "香蕉", "橙子")
    list_fruits = list(tuple_fruits)
    print(f"元组转列表: {list_fruits}")
    
    # 展示不可变性的优势
    print("\n元组的不可变性优势:")
    try:
        # 尝试修改元组中的列表
        nested_tuple = ([1, 2], [3, 4])
        nested_tuple[0].append(5)  # 可以修改内部列表
        print(f"修改内部列表后: {nested_tuple}")
        
        nested_tuple[0] = [6, 7]  # 不能修改元组本身
    except TypeError as e:
        print(f"修改元组本身的错误: {e}")

if __name__ == "__main__":
    basic_tuple_operations()
    tuple_usage()
    tuple_methods()
    tuple_list_conversion() 