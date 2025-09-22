import ray
import torch
import torch.nn as nn
import socket
import os
import time
import logging

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查设备数量
print(f"Device count: {torch.cuda.device_count()}")

class ColumnTPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        device_count = torch.cuda.device_count()
        rank = os.environ['RANK']
        print(f"device_count: {device_count}, rank: {rank}")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=int(rank))

        self.layer = nn.Linear(input_size, output_size, bias=False).cuda(0)

    def forward(self, x):
        ret = self.layer(x.cuda(0))
        return ret

    def load_weights(self, weight):
        # 所有worker都加载完整的权重矩阵，而不是分片
        self.layer.weight.data.copy_(weight)

    def state_dict(self):
        return self.layer.state_dict()

ray.init()

master_addr = ray._private.services.get_node_ip_address()
with socket.socket() as sock:
    sock.bind(('', 0))
    master_port = sock.getsockname()[1]

num_gpus = 2
workers = []

for i in range(num_gpus):

    options = {'runtime_env': {'env_vars': {'WORLD_SIZE': str(num_gpus), 'RANK': str(i), 'MASTER_ADDR': master_addr, 'MASTER_PORT': str(master_port)}}}

    workers.append(ray.remote(num_gpus=1)(ColumnTPLayer).options(**options).remote(2, 6))

batch_size = 10
input_data = torch.randn(batch_size, 2)

full_layer = torch.nn.Linear(2, 6, bias=False)
weight = full_layer.state_dict()['weight']

ret_list = []
for i in range(num_gpus):
    _ = ray.get(workers[i].load_weights.remote(weight))

for i in range(num_gpus):
    ret_list.append(workers[i].forward.remote(input_data))

ret = ray.get(ret_list)

ray.shutdown()

full_layer = full_layer.cuda()
fl_ret = full_layer(input_data.cuda()).cpu()

torch.testing.assert_close(ret[0], ret[1])
torch.testing.assert_close(ret[0].cpu(), fl_ret)