import ray
import time

class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value



def main():
    ray.init()

    CounterActor = ray.remote(Counter)
    counter = CounterActor.remote()
    
    futures = [counter.increment.remote() for _ in range(10)]
    results = ray.get(futures)

    final_value = ray.get(counter.get_value.remote())

    print(f"计数器递增过程: {results}")
    print(f"计数器最终值: {final_value}")
    print(f"计数器递增次数: {len(results)}")

    ray.shutdown()




if __name__ == "__main__":
    main()
