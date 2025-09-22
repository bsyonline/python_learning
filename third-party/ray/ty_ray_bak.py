import ray
import torch
import torch.nn as nn
import socket
import os
import time
import logging

# Initialize distributed training for Ray workers
def init_distributed_training():
    if not torch.distributed.is_initialized():
        # Get distributed training parameters from environment variables
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12345')
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        
        # Check if CUDA is available and if the requested device exists
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            torch.distributed.init_process_group(backend="nccl", 
                                                init_method=f"tcp://{master_addr}:{master_port}", 
                                                world_size=world_size, 
                                                rank=rank)
            torch.cuda.set_device(rank)
        elif torch.cuda.is_available():
            # If CUDA is available but requested device doesn't exist, use device 0
            torch.distributed.init_process_group(backend="nccl", 
                                                init_method=f"tcp://{master_addr}:{master_port}", 
                                                world_size=world_size, 
                                                rank=rank)
            torch.cuda.set_device(0)  # Fallback to device 0
            logging.warning(f"Requested GPU {rank} not available. Using GPU 0 instead.")
        else:
            # If CUDA is not available, initialize with gloo backend for CPU-only training
            torch.distributed.init_process_group(backend="gloo", 
                                                init_method=f"tcp://{master_addr}:{master_port}", 
                                                world_size=world_size, 
                                                rank=rank)

class ColumnTPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Initialize distributed training
        init_distributed_training()
        
        # 根据world_size选择是否进行切分
        world_size = int(os.environ['WORLD_SIZE'])
        layer_output_size = output_size // world_size if world_size > 1 else output_size
        self.layer = nn.Linear(input_size, layer_output_size, bias=False).cuda()

    def forward(self, x):
        # 确保输入数据在正确的设备上
        rank = int(os.environ['RANK'])
        
        # Check if CUDA is available and if the requested device exists
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            device = torch.device(f'cuda:{rank}')
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')  # Fallback to device 0
        else:
            device = torch.device('cpu')
            
        x = x.to(device)
        ret = self.layer(x)
        
        # 只有当world_size大于1时才执行分布式操作
        world_size = int(os.environ['WORLD_SIZE'])
        if world_size > 1:
            output_tensor = torch.zeros(size=(world_size, ret.shape[0], ret.shape[1]), dtype=ret.dtype, device=ret.device)
            torch.distributed.all_gather_into_tensor(output_tensor, ret, async_op=False)
            output_tensor = torch.cat(output_tensor.unbind(dim=0), dim=-1)
            return output_tensor
        else:
            # 当world_size=1时，直接返回本地计算结果
            return ret

    def load_weights(self, weight):
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Check if CUDA is available and if the requested device exists
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            device = torch.device(f'cuda:{rank}')
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')  # Fallback to device 0
        else:
            device = torch.device('cpu')
        
        # 只有当world_size大于1时才进行权重切分
        if world_size > 1:
            dim_per_rank = weight.shape[0] // world_size
            self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :].to(device))
        else:
            # 当world_size=1时，直接复制全部权重
            self.layer.weight.data.copy_(weight.to(device))

    def state_dict(self):
        return self.layer.state_dict()

try:
    ray.init()

    # Get master address and port
    master_addr = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(('', 0))
        master_port = sock.getsockname()[1]

    # Check available GPUs and adjust number of workers accordingly
    num_gpus = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    print(f"Using {num_gpus} GPUs for training")
    workers = []

    for i in range(num_gpus):
        options = {'runtime_env': {'env_vars': {'WORLD_SIZE': str(num_gpus), 'RANK': str(i), 'MASTER_ADDR': master_addr, 'MASTER_PORT': str(master_port)}}}
        workers.append(ray.remote(num_gpus=1)(ColumnTPLayer).options(**options).remote(2, 6))

    batch_size = 10
    input_data = torch.randn(batch_size, 2)

    full_layer = torch.nn.Linear(2, 6, bias=False)
    weight = full_layer.state_dict()['weight']

    # Load weights for each worker
    ret_list = []
    for i in range(num_gpus):
        _ = ray.get(workers[i].load_weights.remote(weight))

    # Forward pass for each worker
    for i in range(num_gpus):
        ret_list.append(workers[i].forward.remote(input_data))

    ret = ray.get(ret_list)

    ray.shutdown()

    # Ensure full_layer runs on the correct device based on available GPUs
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    full_layer = full_layer.to(device)
    input_data = input_data.to(device)
    fl_ret = full_layer(input_data).cpu()

    torch.testing.assert_close(ret[0], ret[1])
    torch.testing.assert_close(ret[0].cpu(), fl_ret)
finally:
    # Clean up distributed process group if initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()