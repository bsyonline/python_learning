import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. 稀疏张量基础操作
print("1. 稀疏张量基础操作:")
# 创建一个普通的密集张量
dense_tensor = torch.zeros(5, 6)
dense_tensor[0, 1] = 1.0
dense_tensor[2, 3] = 2.0
dense_tensor[4, 5] = 3.0

print("密集张量:")
print(dense_tensor)

# 转换为COO格式的稀疏张量
indices = torch.nonzero(dense_tensor).t()
values = dense_tensor[indices[0], indices[1]]
sparse_tensor = torch.sparse_coo_tensor(indices, values, dense_tensor.size())

print("\n稀疏张量:")
print("索引:", indices)
print("值:", values)
print("形状:", sparse_tensor.size())
print("稀疏度:", 1.0 - len(values) / dense_tensor.numel())

# 2. 嵌入层的稀疏版本
print("\n2. 稀疏嵌入层示例:")
num_embeddings = 1000
embedding_dim = 10
# 创建稀疏嵌入层
sparse_embedding = nn.Embedding(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    sparse=True  # 启用稀疏梯度
)

# 创建输入索引
input_indices = torch.LongTensor([1, 5, 3, 7])
output = sparse_embedding(input_indices)
print("输入索引形状:", input_indices.shape)
print("输出嵌入形状:", output.shape)

# 3. 稀疏线性层
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # 创建权重矩阵
        weight = torch.randn(out_features, in_features)
        # 随机将一部分权重置为0以创建稀疏性
        mask = torch.rand_like(weight) > self.sparsity
        self.weight = nn.Parameter(weight * mask.float())
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # 将权重转换为稀疏张量
        indices = torch.nonzero(self.weight).t()
        values = self.weight[indices[0], indices[1]]
        sparse_weight = torch.sparse_coo_tensor(
            indices, values, self.weight.size()
        )
        # 使用稀疏矩阵乘法
        return torch.sparse.mm(sparse_weight, x.t()).t() + self.bias

print("\n3. 稀疏线性层示例:")
sparse_linear = SparseLinear(20, 10, sparsity=0.7)
input_data = torch.randn(5, 20)
output = sparse_linear(input_data)
print("输入形状:", input_data.shape)
print("输出形状:", output.shape)
print("权重稀疏度:", 1.0 - torch.count_nonzero(sparse_linear.weight) / sparse_linear.weight.numel())

# 4. 稀疏注意力机制
def sparse_attention(query, key, value, sparsity=0.5):
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1))
    # 创建稀疏掩码
    mask = torch.rand_like(scores) > sparsity
    scores = scores.masked_fill(~mask, float('-inf'))
    # 应用softmax
    attention_weights = F.softmax(scores, dim=-1)
    # 计算输出
    return torch.matmul(attention_weights, value)

print("\n4. 稀疏注意力机制示例:")
seq_len = 10
hidden_dim = 8
query = torch.randn(2, seq_len, hidden_dim)
key = torch.randn(2, seq_len, hidden_dim)
value = torch.randn(2, seq_len, hidden_dim)

sparse_attn_output = sparse_attention(query, key, value, sparsity=0.7)
print("注意力输出形状:", sparse_attn_output.shape)

# 5. 块稀疏矩阵
class BlockSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size=4, sparsity=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.sparsity = sparsity
        
        # 确保特征数能被块大小整除
        assert in_features % block_size == 0 and out_features % block_size == 0
        
        # 创建块稀疏权重
        num_blocks_in = in_features // block_size
        num_blocks_out = out_features // block_size
        block_mask = torch.rand(num_blocks_out, num_blocks_in) > sparsity
        
        # 展开块掩码到完整权重
        mask = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
        weight = torch.randn(out_features, in_features)
        self.weight = nn.Parameter(weight * mask.float())
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

print("\n5. 块稀疏线性层示例:")
block_sparse = BlockSparseLinear(16, 12, block_size=4, sparsity=0.5)
input_data = torch.randn(3, 16)
output = block_sparse(input_data)
print("输入形状:", input_data.shape)
print("输出形状:", output.shape)
print("权重稀疏度:", 1.0 - torch.count_nonzero(block_sparse.weight) / block_sparse.weight.numel())

if __name__ == "__main__":
    # 实际应用示例
    print("\n实际应用示例:")
    
    # 1. 大规模词嵌入
    print("\n1. 大规模词嵌入示例:")
    vocab_size = 100000
    embedding_dim = 128
    sparse_emb = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    
    # 模拟批处理输入
    batch_indices = torch.randint(0, vocab_size, (32, 10))
    emb_output = sparse_emb(batch_indices)
    print(f"词嵌入输出形状: {emb_output.shape}")
    
    # 2. 稀疏分类器
    print("\n2. 稀疏分类器示例:")
    class SparseClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, sparsity=0.7):
            super().__init__()
            self.sparse_linear1 = SparseLinear(input_size, hidden_size, sparsity)
            self.sparse_linear2 = SparseLinear(hidden_size, num_classes, sparsity)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.sparse_linear1(x)
            x = self.relu(x)
            x = self.sparse_linear2(x)
            return x
    
    classifier = SparseClassifier(100, 50, 10)
    sample_input = torch.randn(16, 100)
    output = classifier(sample_input)
    print(f"分类器输出形状: {output.shape}")
    
    # 稀疏层的优点和注意事项
    print("\n稀疏层的优点:")
    print("1. 内存效率")
    print("   - 只存储非零元素")
    print("   - 适用于大规模模型")
    
    print("\n2. 计算效率")
    print("   - 减少浮点运算")
    print("   - 可能提高推理速度")
    
    print("\n3. 正则化效果")
    print("   - 减少过拟合")
    print("   - 模型压缩")
    
    print("\n注意事项:")
    print("1. 稀疏度选择")
    print("   - 需要平衡精度和效率")
    print("   - 通常从50%-90%开始尝试")
    
    print("\n2. 训练考虑")
    print("   - 可能需要特殊的优化器")
    print("   - 梯度更新可能不稳定")
    
    print("\n3. 硬件支持")
    print("   - 需要检查硬件对稀疏运算的支持")
    print("   - 某些平台可能无法充分利用稀疏性")
    
    # 性能测试
    print("\n性能测试:")
    import time
    
    # 测试稀疏vs密集
    input_size = 1000
    output_size = 100
    batch_size = 32
    
    # 密集层
    dense_layer = nn.Linear(input_size, output_size)
    # 稀疏层
    sparse_layer = SparseLinear(input_size, output_size, sparsity=0.9)
    
    test_input = torch.randn(batch_size, input_size)
    
    # 测试密集层
    start_time = time.time()
    _ = dense_layer(test_input)
    dense_time = time.time() - start_time
    
    # 测试稀疏层
    start_time = time.time()
    _ = sparse_layer(test_input)
    sparse_time = time.time() - start_time
    
    print(f"密集层处理时间: {dense_time*1000:.2f}ms")
    print(f"稀疏层处理时间: {sparse_time*1000:.2f}ms")
    print(f"内存使用比较:")
    print(f"密集层参数数量: {sum(p.numel() for p in dense_layer.parameters())}")
    print(f"稀疏层非零参数数量: {torch.count_nonzero(sparse_layer.weight).item()}") 