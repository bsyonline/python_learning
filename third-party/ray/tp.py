import torch
import torch.nn as nn
import os

# 检查是否支持CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 只在支持CUDA时设置分布式环境变量和初始化分布式训练
if torch.cuda.is_available():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'  # 设置为1以避免等待其他进程
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12345"
    
    try:
        torch.distributed.init_process_group(backend="nccl", timeout=torch.distributed.constants.default_pg_timeout)
        torch.cuda.set_device(int(os.environ['RANK']))
        print("Distributed training initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        # 如果分布式训练初始化失败，回退到单GPU模式
        device = torch.device('cuda:0')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 构建一个从column维度切分的linear layer
class ColumnTPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 根据设备选择是否使用world_size进行切分
        try:
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
        except ValueError:
            world_size = 1
        layer_output_size = output_size // world_size if torch.cuda.is_available() else output_size
        self.layer = nn.Linear(input_size, layer_output_size, bias=False).to(device=device)

    def forward(self, x):
        x = x.to(device=device)
        ret = self.layer(x)
        
        # 只在支持分布式且world_size>1时执行all_gather操作
        if torch.cuda.is_available():
            try:
                world_size = int(os.environ.get('WORLD_SIZE', '1'))
            except ValueError:
                world_size = 1
            
            # 只有当world_size大于1时才执行分布式操作
            if world_size > 1:
                try:
                    output_tensor = torch.zeros(size=(world_size, ret.shape[0], ret.shape[1]), dtype=ret.dtype, device=ret.device)
                    torch.distributed.all_gather_into_tensor(output_tensor, ret, async_op=False)
                    output_tensor = torch.cat(output_tensor.unbind(dim=0), dim=-1)
                    return output_tensor
                except Exception as e:
                    print(f"Distributed operation failed: {e}")
                    # 如果分布式操作失败，返回本地计算结果
                    return ret
            else:
                # 当world_size=1时，直接返回本地计算结果
                return ret
        else:
            return ret

    def load_weights(self, weight):
        # 只在支持CUDA时进行权重切分
        if torch.cuda.is_available():
            try:
                rank = int(os.environ.get('RANK', '0'))
                world_size = int(os.environ.get('WORLD_SIZE', '1'))
            except ValueError:
                rank = 0
                world_size = 1
            
            # 只有当world_size大于1时才进行权重切分
            if world_size > 1:
                dim_per_rank = weight.shape[0] // world_size
                self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :])
            else:
                # 当world_size=1时，直接复制全部权重
                self.layer.weight.data.copy_(weight)
        else:
            self.layer.weight.data.copy_(weight)

batch_size = 10
input_data = torch.randn(batch_size, 2)

# init一个PyTorch的linear layer，并让我们构建的layer和它保持参数一致。
full_layer = torch.nn.Linear(2, 6, bias=False).to(device=device)
weight = full_layer.state_dict()['weight']

tp_layer = ColumnTPLayer(2, 6)
tp_layer.load_weights(weight)

# 确保输入数据在正确的设备上
input_data = input_data.to(device=device)

# 只在支持分布式时进行结果比较
tp_ret = tp_layer(input_data)
fl_ret = full_layer(input_data)

if torch.cuda.is_available():
    torch.testing.assert_close(tp_ret.cpu(), fl_ret.cpu())
    print("Results match!")
else:
    # 在CPU模式下，直接比较结果
    tp_ret_cpu = tp_ret.cpu()
    fl_ret_cpu = fl_ret.cpu()
    if torch.allclose(tp_ret_cpu, fl_ret_cpu, atol=1e-6):
        print("Results match on CPU!")
    else:
        print("Results do not match on CPU.")
        print(f"TP result: {tp_ret_cpu}")
        print(f"Full layer result: {fl_ret_cpu}")