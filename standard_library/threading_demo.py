# Python标准库 - 线程操作

import threading
import time
import queue

# 1. 基本线程创建和使用
def basic_thread():
    print("基本线程示例：")
    
    def worker(name, delay):
        """工作线程函数"""
        print(f"线程 {name} 开始工作")
        time.sleep(delay)
        print(f"线程 {name} 完成工作")
    
    # 创建多个线程
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(f"Thread-{i}", i))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("所有线程已完成")

# 2. 线程同步 - 使用Lock
def thread_synchronization():
    print("\n线程同步示例：")
    
    # 创建一个共享资源
    counter = 0
    counter_lock = threading.Lock()
    
    def increment(name, count):
        nonlocal counter
        for _ in range(count):
            with counter_lock:  # 使用锁保护共享资源
                current = counter
                time.sleep(0.001)  # 模拟一些处理时间
                counter = current + 1
                print(f"线程 {name}: counter = {counter}")
    
    # 创建多个线程同时增加计数器
    threads = []
    for i in range(3):
        t = threading.Thread(target=increment, args=(f"Thread-{i}", 3))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"最终计数: {counter}")

# 3. 线程通信 - 使用Queue
def thread_communication():
    print("\n线程通信示例：")
    
    # 创建一个队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def producer():
        """生产者线程"""
        for i in range(5):
            task = f"Task-{i}"
            task_queue.put(task)
            print(f"生产者生产: {task}")
            time.sleep(0.5)
    
    def consumer():
        """消费者线程"""
        while True:
            try:
                task = task_queue.get(timeout=2)
                print(f"消费者处理: {task}")
                result = f"Result-{task}"
                result_queue.put(result)
                task_queue.task_done()
                time.sleep(1)
            except queue.Empty:
                break
    
    # 创建生产者和消费者线程
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    
    producer_thread.start()
    consumer_thread.start()
    
    producer_thread.join()
    consumer_thread.join()
    
    # 获取所有结果
    print("\n处理结果:")
    while not result_queue.empty():
        print(result_queue.get())

# 4. 线程池
def thread_pool():
    print("\n线程池示例：")
    
    from concurrent.futures import ThreadPoolExecutor
    
    def worker(x):
        """工作函数"""
        time.sleep(1)
        return x * x
    
    # 创建线程池
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交任务
        futures = [executor.submit(worker, i) for i in range(5)]
        
        # 获取结果
        for future in futures:
            print(f"结果: {future.result()}")

if __name__ == "__main__":
    basic_thread()
    thread_synchronization()
    thread_communication()
    thread_pool() 