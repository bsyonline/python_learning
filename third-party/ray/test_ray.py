import ray
import time

@ray.remote
def compute_square(x):
    time.sleep(0.1)  # 模拟计算时间
    return x * x

def main():
    ray.init()
    numbers = list(range(1, 11))
    
    # 串行执行（逐个获取结果）
    start_time = time.time()
    serial_results = []
    for x in numbers:
        result = compute_square.remote(x)
        serial_results.append(ray.get(result))  # 逐个获取结果
    serial_time = time.time() - start_time
    print(f"串行计算结果: {serial_results}")
    print(f"串行执行时间: {serial_time:.2f}秒")

    # 并行执行（批量获取结果）
    start_time = time.time()
    parallel_results = [compute_square.remote(x) for x in numbers]
    parallel_results = ray.get(parallel_results)  # 批量获取结果
    parallel_time = time.time() - start_time
    print(f"并行计算结果: {parallel_results}")
    print(f"并行执行时间: {parallel_time:.2f}秒")

    # 关闭Ray
    ray.shutdown()



if __name__ == "__main__":
    main()



