import ray
import time
import numpy as np

# 初始化 Ray
ray.init()

@ray.remote
def generate_data(size):
    """生成随机数据的任务"""
    time.sleep(0.1)  # 模拟计算时间
    return np.random.rand(size)

@ray.remote
def process_data(data):
    """处理数据的任务"""
    time.sleep(0.1)  # 模拟计算时间
    return np.sum(data ** 2)

@ray.remote
def combine_results(*results):
    """合并结果的任务"""
    return np.mean(results)

def main():
    print("Ray 高级功能示例：任务依赖和共享内存对象")
    
    # 1. 使用 Ray.put 将大对象放入对象存储
    print("\n1. 使用 Ray.put 存储大对象")
    large_array = np.random.rand(1000000)  # 创建一个大数组
    start_time = time.time()
    object_ref = ray.put(large_array)  # 将数组放入 Ray 对象存储
    put_time = time.time() - start_time
    print(f"对象放入时间: {put_time:.4f}秒")
    
    # 2. 从对象存储中获取对象
    start_time = time.time()
    retrieved_array = ray.get(object_ref)  # 从对象存储中获取数组
    get_time = time.time() - start_time
    print(f"对象获取时间: {get_time:.4f}秒")
    print(f"数组大小: {len(retrieved_array)}")
    
    # 3. 演示任务依赖
    print("\n2. 演示任务依赖")
    # 生成多个数据集
    data_refs = [generate_data.remote(100000) for _ in range(5)]
    
    # 处理数据集（依赖于前面的任务）
    processed_refs = [process_data.remote(data_ref) for data_ref in data_refs]
    
    # 合并结果（依赖于处理后的数据）
    final_result = ray.get(combine_results.remote(*processed_refs))
    
    print(f"最终合并结果: {final_result}")
    
    # 4. 展示 Ray 的并行处理能力
    print("\n3. 展示并行处理能力")
    start_time = time.time()
    # 并行执行多个任务
    futures = [generate_data.remote(100000) for _ in range(10)]
    results = ray.get(futures)
    parallel_time = time.time() - start_time
    print(f"并行执行10个任务时间: {parallel_time:.2f}秒")
    
    # 关闭 Ray
    ray.shutdown()

if __name__ == "__main__":
    main()