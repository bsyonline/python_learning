# Python设计模式 - 迭代器模式
# 提供一种方法顺序访问一个聚合对象中的各个元素，而又不需要暴露该对象的内部表示

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Iterator
from collections.abc import Iterable
from datetime import datetime, timedelta

# 1. 自定义迭代器接口
class CustomIterator(ABC):
    """自定义迭代器接口"""
    
    @abstractmethod
    def has_next(self) -> bool:
        """是否还有下一个元素"""
        pass
    
    @abstractmethod
    def next(self) -> Any:
        """获取下一个元素"""
        pass
    
    @abstractmethod
    def current(self) -> Any:
        """获取当前元素"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置迭代器"""
        pass

# 2. 具体迭代器
class ArrayIterator(CustomIterator):
    """数组迭代器"""
    
    def __init__(self, collection: List[Any]):
        self._collection = collection
        self._position = 0
    
    def has_next(self) -> bool:
        return self._position < len(self._collection)
    
    def next(self) -> Any:
        if self.has_next():
            item = self._collection[self._position]
            self._position += 1
            return item
        raise StopIteration()
    
    def current(self) -> Any:
        if 0 <= self._position < len(self._collection):
            return self._collection[self._position]
        raise ValueError("当前位置无效")
    
    def reset(self) -> None:
        self._position = 0

# 3. Python内置迭代器实现
class DateRange(Iterable):
    """日期范围类"""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
    
    def __iter__(self) -> Iterator[datetime]:
        current_date = self.start_date
        while current_date <= self.end_date:
            yield current_date
            current_date += timedelta(days=1)
    
    def reverse(self) -> Iterator[datetime]:
        """反向迭代器"""
        current_date = self.end_date
        while current_date >= self.start_date:
            yield current_date
            current_date -= timedelta(days=1)

# 4. 树形结构迭代器
class TreeNode:
    """树节点"""
    
    def __init__(self, value: Any):
        self.value = value
        self.children: List[TreeNode] = []
    
    def add_child(self, child: 'TreeNode') -> None:
        """添加子节点"""
        self.children.append(child)

class TreeIterator(CustomIterator):
    """树形结构迭代器 - 深度优先遍历"""
    
    def __init__(self, root: TreeNode):
        self._root = root
        self._stack: List[TreeNode] = []
        self._current: Optional[TreeNode] = None
        self.reset()
    
    def has_next(self) -> bool:
        return len(self._stack) > 0
    
    def next(self) -> Any:
        if not self.has_next():
            raise StopIteration()
        
        self._current = self._stack.pop()
        # 将子节点按反序入栈（确保正序遍历）
        for child in reversed(self._current.children):
            self._stack.append(child)
        
        return self._current.value
    
    def current(self) -> Any:
        if self._current is None:
            raise ValueError("迭代器未启动")
        return self._current.value
    
    def reset(self) -> None:
        self._stack = [self._root]
        self._current = None

# 5. 分页迭代器
class PageIterator(CustomIterator):
    """分页迭代器"""
    
    def __init__(self, items: List[Any], page_size: int):
        self._items = items
        self._page_size = page_size
        self._current_page = 0
        self._total_pages = (len(items) + page_size - 1) // page_size
    
    def has_next(self) -> bool:
        return self._current_page < self._total_pages
    
    def next(self) -> List[Any]:
        if not self.has_next():
            raise StopIteration()
        
        start_idx = self._current_page * self._page_size
        end_idx = min(start_idx + self._page_size, len(self._items))
        page = self._items[start_idx:end_idx]
        
        self._current_page += 1
        return page
    
    def current(self) -> List[Any]:
        if self._current_page == 0:
            raise ValueError("迭代器未启动")
        
        start_idx = (self._current_page - 1) * self._page_size
        end_idx = min(start_idx + self._page_size, len(self._items))
        return self._items[start_idx:end_idx]
    
    def reset(self) -> None:
        self._current_page = 0

# 6. 使用示例
def iterator_demo():
    print("迭代器模式示例：")
    
    # 数组迭代器示例
    print("\n1. 数组迭代器示例:")
    numbers = [1, 2, 3, 4, 5]
    array_iterator = ArrayIterator(numbers)
    
    print("正向遍历:")
    while array_iterator.has_next():
        print(array_iterator.next(), end=" ")
    
    # 日期范围迭代器示例
    print("\n\n2. 日期范围迭代器示例:")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 5)
    date_range = DateRange(start, end)
    
    print("正向遍历日期:")
    for date in date_range:
        print(date.strftime("%Y-%m-%d"), end=" ")
    
    print("\n反向遍历日期:")
    for date in date_range.reverse():
        print(date.strftime("%Y-%m-%d"), end=" ")
    
    # 树形结构迭代器示例
    print("\n\n3. 树形结构迭代器示例:")
    root = TreeNode("A")
    b_node = TreeNode("B")
    c_node = TreeNode("C")
    root.add_child(b_node)
    root.add_child(c_node)
    b_node.add_child(TreeNode("D"))
    b_node.add_child(TreeNode("E"))
    c_node.add_child(TreeNode("F"))
    
    tree_iterator = TreeIterator(root)
    print("深度优先遍历:")
    while tree_iterator.has_next():
        print(tree_iterator.next(), end=" ")
    
    # 分页迭代器示例
    print("\n\n4. 分页迭代器示例:")
    items = list(range(1, 11))  # 1-10的数字
    page_iterator = PageIterator(items, page_size=3)
    
    print("按页遍历:")
    page_num = 1
    while page_iterator.has_next():
        page = page_iterator.next()
        print(f"第{page_num}页: {page}")
        page_num += 1

if __name__ == "__main__":
    iterator_demo() 