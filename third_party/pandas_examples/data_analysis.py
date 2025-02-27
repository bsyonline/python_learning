# Pandas数据分析示例

import pandas as pd
import numpy as np

# 1. 创建数据结构
def create_data_structures():
    print("创建数据结构示例：")
    
    # 创建Series
    series = pd.Series([1, 3, 5, np.nan, 6, 8])
    print(f"Series示例:\n{series}")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    })
    print(f"\nDataFrame示例:\n{df}")

# 2. 数据操作
def data_operations():
    print("\n数据操作示例：")
    
    # 创建示例数据
    df = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [25, 30, 35, 40],
        '城市': ['北京', '上海', '广州', '深圳'],
        '薪资': [10000, 20000, 15000, 25000]
    })
    print(f"原始数据:\n{df}")
    
    # 选择数据
    print(f"\n选择'姓名'列:\n{df['姓名']}")
    print(f"\n选择前两行:\n{df.head(2)}")
    
    # 条件筛选
    high_salary = df[df['薪资'] > 15000]
    print(f"\n高薪资员工:\n{high_salary}")
    
    # 添加新列
    df['年薪'] = df['薪资'] * 12
    print(f"\n添加年薪后:\n{df}")

# 3. 数据统计和聚合
def statistics_aggregation():
    print("\n数据统计和聚合示例：")
    
    # 创建示例数据
    df = pd.DataFrame({
        '部门': ['技术', '销售', '技术', '销售', '市场'],
        '薪资': [12000, 15000, 18000, 20000, 16000],
        '年龄': [25, 30, 28, 35, 32]
    })
    print(f"原始数据:\n{df}")
    
    # 基本统计
    print(f"\n基本统计信息:\n{df.describe()}")
    
    # 分组统计
    group_stats = df.groupby('部门').agg({
        '薪资': ['mean', 'min', 'max'],
        '年龄': 'mean'
    })
    print(f"\n按部门分组统计:\n{group_stats}")

# 4. 数据清洗和处理
def data_cleaning():
    print("\n数据清洗和处理示例：")
    
    # 创建包含缺失值的数据
    df = pd.DataFrame({
        '姓名': ['张三', '李四', None, '赵六'],
        '年龄': [25, None, 35, 40],
        '城市': ['北京', '上海', None, '深圳']
    })
    print(f"原始数据（包含缺失值）:\n{df}")
    
    # 处理缺失值
    df_cleaned = df.fillna({
        '姓名': '未知',
        '年龄': df['年龄'].mean(),
        '城市': '未知'
    })
    print(f"\n处理缺失值后:\n{df_cleaned}")
    
    # 删除重复行
    df_no_duplicates = df_cleaned.drop_duplicates()
    print(f"\n删除重复行后:\n{df_no_duplicates}")

if __name__ == "__main__":
    create_data_structures()
    data_operations()
    statistics_aggregation()
    data_cleaning() 