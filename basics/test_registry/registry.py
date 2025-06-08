import importlib
import inspect

class Registry(object):
    @staticmethod
    def register(module_path, new_func):
        # 分割模块路径和类名/函数名
        parts = module_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError("模块路径格式错误，应包含模块名和类名/函数名")
        module_and_class, func_name = parts
        
        # 检查是否有类名
        if '.' in module_and_class:
            # 尝试解析为类方法，如 foo.operator.Operator.add
            module_name, class_name = module_and_class.rsplit('.', 1)
            
            try:
                # 尝试导入模块
                module = importlib.import_module(module_name)
                
                # 检查class_name是否存在于模块中
                if hasattr(module, class_name):
                    # 检查是否是类
                    target_attr = getattr(module, class_name)
                    if inspect.isclass(target_attr):
                        # 是类，按类方法处理
                        target_class = target_attr
                        
                        # 创建一个适配器函数，忽略self参数
                        def adapter(self, *args, **kwargs):
                            return new_func(*args, **kwargs)
                            
                        setattr(target_class, func_name, adapter)
                        return
                    
                # 如果不是类或者不存在，尝试作为模块处理
                full_module_name = module_and_class
                try:
                    module = importlib.import_module(full_module_name)
                    setattr(module, func_name, new_func)
                    return
                except ImportError:
                    # 既不是类也不是模块，抛出错误
                    raise ValueError(f"{module_and_class} 既不是有效的类也不是有效的模块")
                
            except ImportError:
                # 尝试作为模块处理
                try:
                    module = importlib.import_module(module_and_class)
                    setattr(module, func_name, new_func)
                except ImportError:
                    raise ValueError(f"无法导入模块 {module_and_class}")
        else:
            # 处理模块函数，如 foo.operator.multiply
            module_name = module_and_class
            module = importlib.import_module(module_name)
            setattr(module, func_name, new_func)

    @staticmethod
    def create_wrapped_add(add_func):
        def wrapped_add(self):
            return add_func(self.a, self.b)
        return wrapped_add
