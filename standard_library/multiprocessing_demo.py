# Python标准库 - 进程操作

import multiprocessing as mp
import time
import os

# 1. 基本进程创建和使用
def basic_process():
    print("基本进程示例：")
    
    def worker(name):
        """工作进程函数"""
        print(f"进程 {name} (PID: {os.getpid()}) 开始工作")
        time.sleep(2)
        print(f"进程 {name} 完成工作")
    
    # 创建多个进程
    processes = []
    for i in range(3):
        p = mp.Process(target=worker, args=(f"Process-{i}",))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("所有进程已完成")

# 2. 进程间通信 - 使用Queue
def process_communication():
    print("\n进程间通信示例：")
    
    def producer(queue):
        """生产者进程"""
        for i in range(5):
            data = f"Data-{i}"
            queue.put(data)
            print(f"生产者生产: {data}")
            time.sleep(1)
    
    def consumer(queue):
        """消费者进程"""
        while True:
            try:
                data = queue.get(timeout=3)
                print(f"消费者消费: {data}")
                time.sleep(0.5)
            except:
                break
    
    # 创建队列和进程
    queue = mp.Queue()
    p1 = mp.Process(target=producer, args=(queue,))
    p2 = mp.Process(target=consumer, args=(queue,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()

# 3. 进程池
def process_pool():
    print("\n进程池示例：")
    
    def worker(x):
        """工作函数"""
        time.sleep(1)
        return x * x
    
    # 创建进程池
    with mp.Pool(processes=3) as pool:
        # 使用map
        results = pool.map(worker, range(5))
        print(f"Map结果: {results}")
        
        # 使用apply_async
        async_results = [pool.apply_async(worker, (i,)) for i in range(5)]
        for result in async_results:
            print(f"Async结果: {result.get()}")

# 4. 共享内存
def shared_memory():
    print("\n共享内存示例：")
    
    def modifier(array, lock):
        """修改共享内存的进程"""
        with lock:
            print(f"修改前: {list(array)}")
            for i in range(len(array)):
                array[i] *= 2
            print(f"修改后: {list(array)}")
    
    # 创建共享内存数组
    shared_array = mp.Array('i', [1, 2, 3, 4, 5])
    lock = mp.Lock()
    
    # 创建多个进程修改共享内存
    processes = []
    for i in range(2):
        p = mp.Process(target=modifier, args=(shared_array, lock))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print(f"最终结果: {list(shared_array)}")

if __name__ == "__main__":
    basic_process()
    process_communication()
    process_pool()
    shared_memory() 