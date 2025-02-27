# Python数据库操作 - 高级特性示例

import sqlite3
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
from orm_demo import Base, Employee, Department, DatabaseOperations

# 1. 数据库连接池
def connection_pool_demo():
    print("数据库连接池示例：")
    
    # 创建带连接池的引擎
    engine = create_engine('sqlite:///company.db', 
                          pool_size=5,
                          max_overflow=10,
                          pool_timeout=30,
                          pool_recycle=3600)
    
    # 创建会话工厂
    Session = sessionmaker(bind=engine)
    
    try:
        # 使用多个会话
        sessions = [Session() for _ in range(3)]
        
        # 执行查询
        for i, session in enumerate(sessions):
            result = session.query(Employee).first()
            print(f"会话 {i+1} 查询结果: {result}")
            
        # 关闭会话
        for session in sessions:
            session.close()
            
    finally:
        # 处理引擎清理
        engine.dispose()

# 2. 批量数据处理
def bulk_data_processing():
    print("\n批量数据处理示例：")
    
    db = DatabaseOperations('sqlite:///company.db')
    try:
        # 批量插入
        employees = [
            Employee(name=f"员工{i}", 
                    salary=5000 + i * 100) 
            for i in range(1000)
        ]
        
        # 使用bulk_save_objects进行批量插入
        db.session.bulk_save_objects(employees)
        db.session.commit()
        print("批量插入完成")
        
        # 批量更新
        db.session.query(Employee)\
            .filter(Employee.salary < 6000)\
            .update({Employee.salary: Employee.salary * 1.1})
        db.session.commit()
        print("批量更新完成")
        
    finally:
        db.close()

# 3. 数据导入导出
def data_import_export():
    print("\n数据导入导出示例：")
    
    # 导出到CSV
    def export_to_csv():
        db = DatabaseOperations('sqlite:///company.db')
        try:
            # 查询数据
            query = db.session.query(Employee).join(Department)
            
            # 转换为DataFrame
            df = pd.read_sql(query.statement, db.session.bind)
            
            # 保存到CSV
            df.to_csv('employees.csv', index=False)
            print("数据已导出到 employees.csv")
            
        finally:
            db.close()
    
    # 从CSV导入
    def import_from_csv():
        db = DatabaseOperations('sqlite:///company.db')
        try:
            # 读取CSV
            df = pd.read_csv('employees.csv')
            
            # 转换为字典列表
            records = df.to_dict('records')
            
            # 插入数据
            for record in records:
                emp = Employee(**record)
                db.session.add(emp)
            
            db.session.commit()
            print("数据已从CSV导入")
            
        finally:
            db.close()
    
    export_to_csv()
    import_from_csv()

# 4. 数据库监控和统计
def database_monitoring():
    print("\n数据库监控和统计示例：")
    
    db = DatabaseOperations('sqlite:///company.db')
    try:
        # 表统计
        emp_count = db.session.query(func.count(Employee.id)).scalar()
        dept_count = db.session.query(func.count(Department.id)).scalar()
        
        print(f"员工总数: {emp_count}")
        print(f"部门总数: {dept_count}")
        
        # 工资统计
        salary_stats = db.session.query(
            func.min(Employee.salary).label('min_salary'),
            func.max(Employee.salary).label('max_salary'),
            func.avg(Employee.salary).label('avg_salary'),
            func.sum(Employee.salary).label('total_salary')
        ).one()
        
        print("\n工资统计:")
        print(f"最低工资: {salary_stats.min_salary:.2f}")
        print(f"最高工资: {salary_stats.max_salary:.2f}")
        print(f"平均工资: {salary_stats.avg_salary:.2f}")
        print(f"工资总额: {salary_stats.total_salary:.2f}")
        
        # 部门人数统计
        dept_stats = db.session.query(
            Department.name,
            func.count(Employee.id).label('emp_count')
        ).join(Employee).group_by(Department.name).all()
        
        print("\n各部门人数:")
        for dept in dept_stats:
            print(f"{dept.name}: {dept.emp_count}人")
        
    finally:
        db.close()

# 5. 数据库备份和恢复
def database_backup_restore():
    print("\n数据库备份和恢复示例：")
    
    def backup_database():
        # 连接源数据库
        source_conn = sqlite3.connect('company.db')
        # 连接备份数据库
        backup_conn = sqlite3.connect('company_backup.db')
        
        try:
            # 执行备份
            source_conn.backup(backup_conn)
            print("数据库备份完成")
            
        finally:
            source_conn.close()
            backup_conn.close()
    
    def restore_database():
        # 连接备份数据库
        backup_conn = sqlite3.connect('company_backup.db')
        # 连接目标数据库
        target_conn = sqlite3.connect('company_restored.db')
        
        try:
            # 执行恢复
            backup_conn.backup(target_conn)
            print("数据库恢复完成")
            
        finally:
            backup_conn.close()
            target_conn.close()
    
    backup_database()
    restore_database()

if __name__ == "__main__":
    connection_pool_demo()
    bulk_data_processing()
    data_import_export()
    database_monitoring()
    database_backup_restore() 