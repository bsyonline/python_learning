import torch
import torch.nn as nn
import math
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 1. 位置编码 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 2. 多头注意力机制 (Multi-Head Attention)
def demonstrate_multihead_attention():
    print("\n2. 多头注意力机制示例:")
    # 创建多头注意力层
    embed_dim = 256
    num_heads = 8
    mha = nn.MultiheadAttention(embed_dim, num_heads)
    
    # 准备输入数据
    seq_len = 10
    batch_size = 2
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = value = query  # 自注意力机制
    
    # 前向传播
    attn_output, attn_weights = mha(query, key, value)
    print(f"多头注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    return mha

# 3. Transformer编码器层 (TransformerEncoderLayer)
def demonstrate_transformer_encoder():
    print("\n3. Transformer编码器层示例:")
    # 创建编码器层
    d_model = 256
    nhead = 8
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,          # 模型维度
        nhead=nhead,              # 注意力头数
        dim_feedforward=1024,     # 前馈网络维度
        dropout=0.1,              # dropout率
        activation='relu',        # 激活函数
        batch_first=True          # 是否将batch维度放在第一维
    )
    
    # 创建完整的编码器
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
    
    # 准备输入数据
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = encoder(x)
    print(f"编码器输出形状: {output.shape}")
    return encoder

# 4. Transformer解码器层 (TransformerDecoderLayer)
def demonstrate_transformer_decoder():
    print("\n4. Transformer解码器层示例:")
    # 创建解码器层
    d_model = 256
    nhead = 8
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation='relu',
        batch_first=True
    )
    
    # 创建完整的解码器
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
    
    # 准备输入数据
    batch_size = 2
    seq_len = 10
    tgt_len = 8
    memory = torch.randn(batch_size, seq_len, d_model)  # 编码器的输出
    tgt = torch.randn(batch_size, tgt_len, d_model)     # 目标序列
    
    # 前向传播
    output = decoder(tgt, memory)
    print(f"解码器输出形状: {output.shape}")
    return decoder

# 5. 完整的Transformer模型
class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length, vocab_size):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
    
    def forward(self, src, tgt):
        # 源序列嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 目标序列嵌入和位置编码
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # 生成掩码
        src_mask = None  # 源序列不需要掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        
        # Transformer前向传播
        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory)
        
        # 生成最终输出
        output = self.output_layer(output)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

if __name__ == "__main__":
    # 演示各个组件
    print("1. 位置编码示例:")
    pos_encoder = PositionalEncoding(d_model=256, max_len=100)
    sample_input = torch.randn(2, 10, 256)  # [batch_size, seq_len, d_model]
    pos_encoded = pos_encoder(sample_input)
    print(f"位置编码后的形状: {pos_encoded.shape}")
    
    # 演示多头注意力
    mha = demonstrate_multihead_attention()
    
    # 演示编码器
    encoder = demonstrate_transformer_encoder()
    
    # 演示解码器
    decoder = demonstrate_transformer_decoder()
    
    # 演示完整的Transformer模型
    print("\n5. 完整Transformer模型示例:")
    # 模型参数
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 1024
    max_seq_length = 100
    vocab_size = 1000
    
    # 创建模型
    model = SimpleTransformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size
    )
    
    # 准备示例数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))  # 源序列
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))  # 目标序列
    
    # 前向传播
    output = model(src, tgt)
    print(f"Transformer模型输出形状: {output.shape}")
    
    # 打印模型架构信息
    print("\nTransformer架构的主要组件:")
    print("1. 位置编码 (Positional Encoding)")
    print("   - 为序列中的每个位置添加位置信息")
    print("   - 使用正弦和余弦函数生成")
    
    print("\n2. 多头注意力机制 (Multi-Head Attention)")
    print("   - 并行处理多个注意力头")
    print("   - 每个头关注不同的特征子空间")
    
    print("\n3. 编码器 (Encoder)")
    print("   - 包含自注意力层和前馈神经网络")
    print("   - 处理输入序列")
    
    print("\n4. 解码器 (Decoder)")
    print("   - 包含自注意力层和编码器-解码器注意力层")
    print("   - 生成输出序列")
    
    print("\n5. 前馈神经网络 (Feed Forward Network)")
    print("   - 两个线性变换，中间有ReLU激活")
    print("   - 处理注意力机制的输出")
    
    print("\n主要应用场景:")
    print("- 机器翻译")
    print("- 文本生成")
    print("- 序列到序列学习")
    print("- 文档摘要")
    print("- 其他序列转换任务") 