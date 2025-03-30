# Python数据库操作 - SQLAlchemy ORM示例

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# 创建基类
Base = declarative_base()

# 1. 定义模型
class Department(Base):
    __tablename__ = 'departments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200))
    employees = relationship("Employee", back_populates="department")
    
    def __repr__(self):
        return f"<Department(name='{self.name}', location='{self.location}')>"

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    salary = Column(Float)
    department_id = Column(Integer, ForeignKey('departments.id'))
    created_at = Column(DateTime, default=datetime.now)
    
    department = relationship("Department", back_populates="employees")
    
    def __repr__(self):
        return f"<Employee(name='{self.name}', salary={self.salary})>"

# 2. 数据库操作类
class DatabaseOperations:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_department(self, name, location):
        """添加部门"""
        dept = Department(name=name, location=location)
        self.session.add(dept)
        self.session.commit()
        return dept
    
    def add_employee(self, name, salary, department_name):
        """添加员工"""
        dept = self.session.query(Department).filter_by(name=department_name).first()
        if not dept:
            dept = self.add_department(department_name, "默认位置")
        
        emp = Employee(name=name, salary=salary, department=dept)
        self.session.add(emp)
        self.session.commit()
        return emp
    
    def get_all_employees(self):
        """获取所有员工"""
        return self.session.query(Employee).all()
    
    def get_department_employees(self, department_name):
        """获取指定部门的所有员工"""
        return self.session.query(Employee).join(Department)\
            .filter(Department.name == department_name).all()
    
    def update_employee_salary(self, employee_id, new_salary):
        """更新员工工资"""
        emp = self.session.query(Employee).get(employee_id)
        if emp:
            emp.salary = new_salary
            self.session.commit()
            return True
        return False
    
    def delete_employee(self, employee_id):
        """删除员工"""
        emp = self.session.query(Employee).get(employee_id)
        if emp:
            self.session.delete(emp)
            self.session.commit()
            return True
        return False
    
    def close(self):
        """关闭会话"""
        self.session.close()

# 3. 使用示例
def orm_example():
    # 创建数据库操作实例
    db = DatabaseOperations('sqlite:///company.db')
    
    try:
        # 添加部门和员工
        print("添加部门和员工:")
        db.add_department("技术部", "A栋5楼")
        db.add_department("市场部", "B栋3楼")
        
        db.add_employee("张三", 5000, "技术部")
        db.add_employee("李四", 6000, "技术部")
        db.add_employee("王五", 7000, "市场部")
        
        # 查询所有员工
        print("\n所有员工:")
        for emp in db.get_all_employees():
            print(f"姓名: {emp.name}, 工资: {emp.salary}, 部门: {emp.department.name}")
        
        # 查询技术部员工
        print("\n技术部员工:")
        tech_employees = db.get_department_employees("技术部")
        for emp in tech_employees:
            print(f"姓名: {emp.name}, 工资: {emp.salary}")
        
        # 更新工资
        print("\n更新张三的工资:")
        emp = db.get_all_employees()[0]
        db.update_employee_salary(emp.id, 5500)
        print(f"更新后的工资: {emp.salary}")
        
        # 删除员工
        print("\n删除最后一个员工:")
        last_emp = db.get_all_employees()[-1]
        db.delete_employee(last_emp.id)
        print("删除成功")
        
    finally:
        db.close()

if __name__ == "__main__":
    orm_example() 