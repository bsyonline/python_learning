import ray
import time

@ray.remote
def task_a():
    """模拟耗时任务A"""
    print("任务A开始执行")
    time.sleep(2)
    print("任务A执行完成")
    return "A完成"

@ray.remote
def task_b():
    """模拟耗时任务B"""
    print("任务B开始执行")
    time.sleep(2)
    print("任务B执行完成")
    return "B完成"

@ray.remote
def task_c(result_a, result_b):
    """依赖于任务A和任务B的结果"""
    print("任务C开始执行，依赖于A和B的结果")
    time.sleep(1)
    print("任务C执行完成")
    return f"C完成，基于{result_a}和{result_b}"

def task_a_normal():
    """模拟耗时任务A - 普通函数版本"""
    print("任务A开始执行")
    time.sleep(2)
    print("任务A执行完成")
    return "A完成"

def task_b_normal():
    """模拟耗时任务B - 普通函数版本"""
    print("任务B开始执行")
    time.sleep(2)
    print("任务B执行完成")
    return "B完成"

def task_c_normal(result_a, result_b):
    """依赖于任务A和任务B的结果 - 普通函数版本"""
    print("任务C开始执行，依赖于A和B的结果")
    time.sleep(1)
    print("任务C执行完成")
    return f"C完成，基于{result_a}和{result_b}"

def sequential_execution():
    """顺序执行任务"""
    print("=== 顺序执行 ===")
    start_time = time.time()
    
    # 顺序执行任务A和B
    result_a = task_a_normal()  # 耗时2秒
    result_b = task_b_normal()  # 耗时2秒
    
    # 执行任务C
    result_c = task_c_normal(result_a, result_b)  # 耗时1秒
    
    end_time = time.time()
    print(f"顺序执行结果: {result_c}")
    print(f"顺序执行总耗时: {end_time - start_time:.2f}秒\n")

def ray_execution():
    """使用Ray并行执行任务"""
    print("=== Ray并行执行 ===")
    start_time = time.time()
    
    # 并行提交任务A和B
    ref_a = task_a.remote()  # 立即返回，不阻塞
    ref_b = task_b.remote()  # 立即返回，不阻塞
    
    # 任务C依赖于A和B的结果
    ref_c = task_c.remote(ref_a, ref_b)  # 立即返回，不阻塞
    
    # 获取最终结果
    result_c = ray.get(ref_c)
    
    end_time = time.time()
    print(f"Ray执行结果: {result_c}")
    print(f"Ray执行总耗时: {end_time - start_time:.2f}秒\n")

def main():
    # 先演示顺序执行
    sequential_execution()
    
    # 初始化Ray
    ray.init()
    
    # 再演示Ray并行执行
    ray_execution()
    
    # 关闭Ray
    ray.shutdown()

if __name__ == "__main__":
    main()