import ray
import time
import numpy as np

def main():
    ray.init()
    large_array = np.random.rand(1000000)
    start_time = time.time()
    object_ref = ray.put(large_array)
    put_time = time.time() - start_time
    print(f"对象放入时间: {put_time:.4f}秒")

    start_time = time.time()
    retrieved_array = ray.get(object_ref)
    get_time = time.time() - start_time
    print(f"对象获取时间: {get_time:.4f}秒")
    print(f"数组大小: {len(retrieved_array)}")

    ray.shutdown()

if __name__ == "__main__":
    main()

