import ray
import time
import numpy as np

def task1():
    print("任务task1开始执行")
    time.sleep(2)
    print("任务task1执行完成")
    return "ray_task1完成"

def task2():
    print("任务task2开始执行")
    time.sleep(2)
    print("任务task2执行完成")
    return "task2完成"

def task3(result_a, result_b):
    print("任务task3开始执行，依赖于task1和task2的结果")
    time.sleep(1)
    print("任务task3执行完成")
    return f"task3完成，基于{result_a}和{result_b}"

@ray.remote
def ray_task1():
    print("任务ray_task1开始执行")
    time.sleep(2)
    print("任务ray_task1执行完成")
    return "ray_task1完成"

@ray.remote
def ray_task2():
    print("任务ray_task2开始执行")
    time.sleep(2)
    print("任务ray_task2执行完成")
    return "ray_task2完成"

@ray.remote
def ray_task3(result_a, result_b):
    print("任务ray_task3开始执行，依赖于ray_task1和ray_task2的结果")
    time.sleep(1)
    print("任务ray_task3执行完成")
    return f"ray_task3完成，基于{result_a}和{result_b}"

def main():
    ray.init()
    start_time = time.time()

    t1 = task1()
    t2 = task2()
    t3 = task3(t1, t2)
    end_time = time.time()
    print(f"任务task3的结果: {t3}")
    print(f"任务task3的耗时: {end_time - start_time:.2f}秒")



    start_time = time.time()
    t1 = ray_task1.remote()
    t2 = ray_task2.remote()
    t3 = ray_task3.remote(t1, t2)
    result3 = ray.get(t3)
    print(f"任务ray_task3的结果: {result3}")
    end_time = time.time()
    print(f"Ray执行结果: {result3}")
    print(f"Ray执行总耗时: {end_time - start_time:.2f}秒\n")

    ray.shutdown()


if __name__ == "__main__":
    main()
