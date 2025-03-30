# Python数据库操作 - MySQL示例

import mysql.connector
from mysql.connector import Error
from datetime import datetime

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'test_db'
}

# 1. 数据库连接和基本操作
def basic_mysql_operations():
    print("基本MySQL操作示例：")
    
    try:
        # 创建连接
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            salary DECIMAL(10, 2),
            department VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 插入数据
        sql = '''
        INSERT INTO employees (name, salary, department)
        VALUES (%s, %s, %s)
        '''
        values = ('张三', 5000.00, '技术部')
        cursor.execute(sql, values)
        
        # 提交事务
        conn.commit()
        print("数据插入成功")
        
    except Error as e:
        print(f"错误: {e}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 2. 批量操作和事务处理
def batch_operations():
    print("\n批量操作和事务示例：")
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 准备批量数据
        employees_data = [
            ('李四', 6000.00, '市场部'),
            ('王五', 7000.00, '销售部'),
            ('赵六', 5500.00, '技术部')
        ]
        
        # 批量插入
        sql = '''
        INSERT INTO employees (name, salary, department)
        VALUES (%s, %s, %s)
        '''
        cursor.executemany(sql, employees_data)
        
        # 提交事务
        conn.commit()
        print(f"成功插入 {cursor.rowcount} 条记录")
        
    except Error as e:
        print(f"错误: {e}")
        if conn.is_connected():
            conn.rollback()
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 3. 高级查询
def advanced_queries():
    print("\n高级查询示例：")
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  # 返回字典格式的结果
        
        # 分组查询
        cursor.execute('''
        SELECT 
            department,
            COUNT(*) as emp_count,
            AVG(salary) as avg_salary,
            MIN(salary) as min_salary,
            MAX(salary) as max_salary
        FROM employees
        GROUP BY department
        ''')
        
        print("\n部门统计:")
        for row in cursor.fetchall():
            print(f"\n部门: {row['department']}")
            print(f"员工数: {row['emp_count']}")
            print(f"平均工资: {row['avg_salary']:.2f}")
            print(f"最低工资: {row['min_salary']:.2f}")
            print(f"最高工资: {row['max_salary']:.2f}")
        
        # 条件查询
        cursor.execute('''
        SELECT name, salary, department
        FROM employees
        WHERE salary > %s
        ORDER BY salary DESC
        ''', (6000.00,))
        
        print("\n高薪员工:")
        for row in cursor.fetchall():
            print(f"姓名: {row['name']}, 工资: {row['salary']}, 部门: {row['department']}")
        
    except Error as e:
        print(f"错误: {e}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 4. 存储过程
def stored_procedures():
    print("\n存储过程示例：")
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 创建存储过程
        cursor.execute('''
        CREATE PROCEDURE IF NOT EXISTS get_employee_by_department(IN dept_name VARCHAR(100))
        BEGIN
            SELECT * FROM employees WHERE department = dept_name;
        END
        ''')
        
        # 调用存储过程
        cursor.callproc('get_employee_by_department', ('技术部',))
        
        # 获取结果
        for result in cursor.stored_results():
            print("\n技术部员工:")
            for row in result.fetchall():
                print(f"ID: {row[0]}, 姓名: {row[1]}, 工资: {row[2]}")
        
    except Error as e:
        print(f"错误: {e}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    basic_mysql_operations()
    batch_operations()
    advanced_queries()
    stored_procedures() 