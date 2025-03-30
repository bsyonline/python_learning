import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. CosineSimilarity基础使用
print("1. CosineSimilarity基础使用:")
# 创建余弦相似度层
cos = nn.CosineSimilarity(dim=1, eps=1e-8)

# 创建示例向量
v1 = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
v2 = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.1]])

# 计算余弦相似度
similarity = cos(v1, v2)
print("输入向量1形状:", v1.shape)
print("输入向量2形状:", v2.shape)
print("余弦相似度:", similarity)
print("相似度形状:", similarity.shape)

# 2. CosineSimilarity不同维度示例
print("\n2. CosineSimilarity不同维度示例:")
# 创建不同维度的余弦相似度层
cos_dim0 = nn.CosineSimilarity(dim=0)
cos_dim1 = nn.CosineSimilarity(dim=1)
cos_dim2 = nn.CosineSimilarity(dim=2)

# 创建3D张量
t1 = torch.randn(2, 3, 4)
t2 = torch.randn(2, 3, 4)

print("3D张量形状:", t1.shape)
print("dim=0 结果形状:", cos_dim0(t1, t2).shape)
print("dim=1 结果形状:", cos_dim1(t1, t2).shape)
print("dim=2 结果形状:", cos_dim2(t1, t2).shape)

# 3. PairwiseDistance基础使用
print("\n3. PairwiseDistance基础使用:")
# 创建不同p范数的成对距离层
dist_l1 = nn.PairwiseDistance(p=1)    # 曼哈顿距离
dist_l2 = nn.PairwiseDistance(p=2)    # 欧几里得距离
dist_linf = nn.PairwiseDistance(p=float('inf'))  # 无穷范数

# 计算不同距离
d_l1 = dist_l1(v1, v2)
d_l2 = dist_l2(v1, v2)
d_linf = dist_linf(v1, v2)

print("L1距离:", d_l1)
print("L2距离:", d_l2)
print("L∞距离:", d_linf)

# 4. 批量处理示例
print("\n4. 批量处理示例:")
# 创建批量数据
batch_size = 5
x1 = torch.randn(batch_size, 3)
x2 = torch.randn(batch_size, 3)

# 计算批量相似度和距离
batch_sim = cos(x1, x2)
batch_dist = dist_l2(x1, x2)

print("批量输入形状:", x1.shape)
print("批量余弦相似度:", batch_sim)
print("批量成对距离:", batch_dist)

# 5. 实际应用：相似度搜索
class SimilaritySearch:
    def __init__(self, metric='cosine'):
        self.metric = metric
        if metric == 'cosine':
            self.similarity = nn.CosineSimilarity(dim=1)
        else:
            self.similarity = nn.PairwiseDistance(p=2)
        self.database = None
    
    def add_to_database(self, vectors):
        self.database = vectors
    
    def search(self, query, k=1):
        if self.metric == 'cosine':
            # 对于余弦相似度，较大值表示更相似
            scores = self.similarity(self.database, query.expand_as(self.database))
            return torch.topk(scores, k)
        else:
            # 对于距离，较小值表示更相似
            distances = self.similarity(self.database, query.expand_as(self.database))
            return torch.topk(distances, k, largest=False)

# 6. 可视化相似度和距离
def plot_similarity_distance():
    # 创建二维向量
    theta = torch.linspace(0, 2*np.pi, 100)
    x = torch.cos(theta)
    y = torch.sin(theta)
    reference = torch.stack([torch.ones_like(theta), torch.zeros_like(theta)], dim=1)
    vectors = torch.stack([x, y], dim=1)
    
    # 计算不同类型的距离
    cos = nn.CosineSimilarity(dim=1)
    dist_l1 = nn.PairwiseDistance(p=1)     # 曼哈顿距离
    dist_l2 = nn.PairwiseDistance(p=2)     # 欧几里得距离
    dist_inf = nn.PairwiseDistance(p=float('inf'))  # 切比雪夫距离
    
    similarities = cos(vectors, reference)
    distances_l1 = dist_l1(vectors, reference)
    distances_l2 = dist_l2(vectors, reference)
    distances_inf = dist_inf(vectors, reference)
    
    # 绘图
    plt.figure(figsize=(15, 10))
    
    # 余弦相似度
    plt.subplot(221)
    plt.plot(theta, similarities.numpy(), linewidth=2)
    plt.title('Cosine Similarity (余弦相似度)', fontsize=12, pad=10)
    plt.xlabel('Angle (角度)', fontsize=10)
    plt.ylabel('Similarity (相似度)', fontsize=10)
    plt.grid(True)
    
    # 曼哈顿距离
    plt.subplot(222)
    plt.plot(theta, distances_l1.numpy(), linewidth=2)
    plt.title('Manhattan Distance (曼哈顿距离, p=1)', fontsize=12, pad=10)
    plt.xlabel('Angle (角度)', fontsize=10)
    plt.ylabel('Distance (距离)', fontsize=10)
    plt.grid(True)
    
    # 欧几里得距离
    plt.subplot(223)
    plt.plot(theta, distances_l2.numpy(), linewidth=2)
    plt.title('Euclidean Distance (欧几里得距离, p=2)', fontsize=12, pad=10)
    plt.xlabel('Angle (角度)', fontsize=10)
    plt.ylabel('Distance (距离)', fontsize=10)
    plt.grid(True)
    
    # 切比雪夫距离
    plt.subplot(224)
    plt.plot(theta, distances_inf.numpy(), linewidth=2)
    plt.title('Chebyshev Distance (切比雪夫距离, p=∞)', fontsize=12, pad=10)
    plt.xlabel('Angle (角度)', fontsize=10)
    plt.ylabel('Distance (距离)', fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 添加不同距离度量的示例
    print("\n不同类型的PairwiseDistance示例:")
    x = torch.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 2.0, 3.1],
                     [4.0, 5.0, 6.2]])
    
    # 计算不同p值的距离
    dist_l1 = nn.PairwiseDistance(p=1)(x, y)
    dist_l2 = nn.PairwiseDistance(p=2)(x, y)
    dist_inf = nn.PairwiseDistance(p=float('inf'))(x, y)
    
    print("\n同一对向量的不同距离度量:")
    print(f"曼哈顿距离 (p=1): {dist_l1}")
    print(f"欧几里得距离 (p=2): {dist_l2}")
    print(f"切比雪夫距离 (p=∞): {dist_inf}")
    
    # 演示相似度搜索
    print("\n相似度搜索示例:")
    # 创建数据库
    database = torch.randn(100, 5)  # 100个5维向量
    query = torch.randn(5)  # 查询向量
    
    # 使用余弦相似度搜索
    cos_search = SimilaritySearch(metric='cosine')
    cos_search.add_to_database(database)
    scores, indices = cos_search.search(query, k=3)
    print("\n余弦相似度搜索结果:")
    print("最相似的3个向量的相似度:", scores)
    print("对应的索引:", indices)
    
    # 使用欧几里得距离搜索
    dist_search = SimilaritySearch(metric='euclidean')
    dist_search.add_to_database(database)
    distances, indices = dist_search.search(query, k=3)
    print("\n欧几里得距离搜索结果:")
    print("最近的3个向量的距离:", distances)
    print("对应的索引:", indices)
    
    # 可视化相似度和距离
    plot_similarity_distance()
    
    # 性能测试
    print("\n性能测试:")
    large_batch1 = torch.randn(1000, 128)
    large_batch2 = torch.randn(1000, 128)
    
    cos_layer = nn.CosineSimilarity(dim=1)
    dist_layer = nn.PairwiseDistance(p=2)
    
    import time
    
    # 测试余弦相似度
    start_time = time.time()
    _ = cos_layer(large_batch1, large_batch2)
    cos_time = time.time() - start_time
    
    # 测试欧几里得距离
    start_time = time.time()
    _ = dist_layer(large_batch1, large_batch2)
    dist_time = time.time() - start_time
    
    print(f"余弦相似度计算时间: {cos_time*1000:.2f}ms")
    print(f"欧几里得距离计算时间: {dist_time*1000:.2f}ms")
    
    # 使用注意事项
    print("\n使用注意事项:")
    print("1. CosineSimilarity:")
    print("   - 值域范围: [-1, 1]")
    print("   - 1表示方向完全相同")
    print("   - -1表示方向完全相反")
    print("   - 0表示正交")
    
    print("\n2. PairwiseDistance:")
    print("   - 值域范围: [0, ∞)")
    print("   - 0表示完全相同")
    print("   - 支持不同的p范数")
    print("   - 常用p=1（曼哈顿）和p=2（欧几里得）")
    
    print("\n3. 应用场景:")
    print("   - 余弦相似度：文本相似度、推荐系统")
    print("   - 欧几里得距离：图像检索、聚类") 