# 待办事项应用

import json
from datetime import datetime

class TodoList:
    def __init__(self):
        self.tasks = []
        self.load_tasks()
    
    def add_task(self, title, description=""):
        """添加新任务"""
        task = {
            "id": len(self.tasks) + 1,
            "title": title,
            "description": description,
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": None
        }
        self.tasks.append(task)
        self.save_tasks()
        return task
    
    def complete_task(self, task_id):
        """完成任务"""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_tasks()
                return True
        return False
    
    def delete_task(self, task_id):
        """删除任务"""
        for task in self.tasks:
            if task["id"] == task_id:
                self.tasks.remove(task)
                self.save_tasks()
                return True
        return False
    
    def get_all_tasks(self):
        """获取所有任务"""
        return self.tasks
    
    def get_pending_tasks(self):
        """获取待完成的任务"""
        return [task for task in self.tasks if task["status"] == "pending"]
    
    def get_completed_tasks(self):
        """获取已完成的任务"""
        return [task for task in self.tasks if task["status"] == "completed"]
    
    def save_tasks(self):
        """保存任务到文件"""
        with open("tasks.json", "w", encoding="utf-8") as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=4)
    
    def load_tasks(self):
        """从文件加载任务"""
        try:
            with open("tasks.json", "r", encoding="utf-8") as f:
                self.tasks = json.load(f)
        except FileNotFoundError:
            self.tasks = []

def print_task(task):
    """打印任务详情"""
    status = "✓" if task["status"] == "completed" else " "
    print(f"[{status}] {task['id']}. {task['title']}")
    if task["description"]:
        print(f"   描述: {task['description']}")
    print(f"   创建时间: {task['created_at']}")
    if task["completed_at"]:
        print(f"   完成时间: {task['completed_at']}")

def main():
    todo = TodoList()
    
    while True:
        print("\n待办事项管理")
        print("1. 添加任务")
        print("2. 查看所有任务")
        print("3. 查看待完成任务")
        print("4. 查看已完成任务")
        print("5. 完成任务")
        print("6. 删除任务")
        print("7. 退出")
        
        choice = input("请选择操作 (1-7): ")
        
        if choice == "1":
            title = input("请输入任务标题: ")
            description = input("请输入任务描述 (可选): ")
            task = todo.add_task(title, description)
            print("任务已添加:")
            print_task(task)
        
        elif choice == "2":
            tasks = todo.get_all_tasks()
            if tasks:
                print("\n所有任务:")
                for task in tasks:
                    print_task(task)
                    print()
            else:
                print("暂无任务")
        
        elif choice == "3":
            tasks = todo.get_pending_tasks()
            if tasks:
                print("\n待完成任务:")
                for task in tasks:
                    print_task(task)
                    print()
            else:
                print("暂无待完成任务")
        
        elif choice == "4":
            tasks = todo.get_completed_tasks()
            if tasks:
                print("\n已完成任务:")
                for task in tasks:
                    print_task(task)
                    print()
            else:
                print("暂无已完成任务")
        
        elif choice == "5":
            try:
                task_id = int(input("请输入要完成的任务ID: "))
                if todo.complete_task(task_id):
                    print("任务已标记为完成")
                else:
                    print("未找到指定任务")
            except ValueError:
                print("请输入有效的任务ID")
        
        elif choice == "6":
            try:
                task_id = int(input("请输入要删除的任务ID: "))
                if todo.delete_task(task_id):
                    print("任务已删除")
                else:
                    print("未找到指定任务")
            except ValueError:
                print("请输入有效的任务ID")
        
        elif choice == "7":
            print("感谢使用！")
            break
        
        else:
            print("无效的选择")

if __name__ == "__main__":
    main() 