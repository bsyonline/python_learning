# Python数据库操作 - SQLite示例

import sqlite3
from datetime import datetime

# 1. 数据库连接和基本操作
def basic_sqlite_operations():
    print("基本SQLite操作示例：")
    
    # 创建连接
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    
    try:
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            created_at TIMESTAMP
        )
        ''')
        
        # 插入数据
        cursor.execute('''
        INSERT INTO users (name, age, created_at)
        VALUES (?, ?, ?)
        ''', ('张三', 25, datetime.now()))
        
        # 提交事务
        conn.commit()
        print("数据插入成功")
        
        # 查询数据
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()
        print("\n所有用户:")
        for user in users:
            print(f"ID: {user[0]}, 姓名: {user[1]}, 年龄: {user[2]}, 创建时间: {user[3]}")
    
    finally:
        # 关闭连接
        conn.close()

# 2. 批量操作和事务处理
def batch_operations():
    print("\n批量操作和事务示例：")
    
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    
    try:
        # 开始事务
        cursor.execute('BEGIN TRANSACTION')
        
        # 批量插入数据
        users_data = [
            ('李四', 30, datetime.now()),
            ('王五', 35, datetime.now()),
            ('赵六', 40, datetime.now())
        ]
        
        cursor.executemany('''
        INSERT INTO users (name, age, created_at)
        VALUES (?, ?, ?)
        ''', users_data)
        
        # 提交事务
        conn.commit()
        print("批量插入成功")
        
    except Exception as e:
        # 发生错误时回滚
        conn.rollback()
        print(f"错误: {e}")
    
    finally:
        conn.close()

# 3. 高级查询
def advanced_queries():
    print("\n高级查询示例：")
    
    conn = sqlite3.connect('example.db')
    conn.row_factory = sqlite3.Row  # 启用行工厂，使结果可以通过列名访问
    cursor = conn.cursor()
    
    try:
        # 条件查询
        cursor.execute('''
        SELECT * FROM users 
        WHERE age > ? 
        ORDER BY age DESC
        ''', (30,))
        
        print("\n年龄大于30的用户:")
        for row in cursor.fetchall():
            print(f"姓名: {row['name']}, 年龄: {row['age']}")
        
        # 聚合查询
        cursor.execute('''
        SELECT 
            COUNT(*) as user_count,
            AVG(age) as avg_age,
            MIN(age) as min_age,
            MAX(age) as max_age
        FROM users
        ''')
        
        stats = cursor.fetchone()
        print("\n统计信息:")
        print(f"用户总数: {stats['user_count']}")
        print(f"平均年龄: {stats['avg_age']:.1f}")
        print(f"最小年龄: {stats['min_age']}")
        print(f"最大年龄: {stats['max_age']}")
        
    finally:
        conn.close()

# 4. 使用上下文管理器
def using_context_manager():
    print("\n使用上下文管理器示例：")
    
    # 使用with语句自动处理连接的关闭
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        
        # 更新数据
        cursor.execute('''
        UPDATE users 
        SET age = age + 1 
        WHERE name = ?
        ''', ('张三',))
        
        # 删除数据
        cursor.execute('''
        DELETE FROM users 
        WHERE name = ?
        ''', ('赵六',))
        
        # 变更会自动提交
        print("数据更新成功")

if __name__ == "__main__":
    basic_sqlite_operations()
    batch_operations()
    advanced_queries()
    using_context_manager() 