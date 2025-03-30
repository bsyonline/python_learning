import torch
import torch.nn as nn
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# 创建示例数据
batch_size = 3
seq_length = 4
input_size = 5
hidden_size = 6

# 输入数据形状: (batch_size, seq_length, input_size)
input_data = torch.randn(batch_size, seq_length, input_size)
print("输入数据形状:", input_data.shape)

# 1. 简单RNN (RNN)
print("\n1. 简单RNN示例:")
rnn = nn.RNN(
    input_size=input_size,    # 输入特征维度
    hidden_size=hidden_size,  # 隐藏状态维度
    num_layers=1,            # RNN层数
    batch_first=True         # 是否将batch维度放在第一维
)

# 前向传播
rnn_output, rnn_hidden = rnn(input_data)
print("RNN输出形状:", rnn_output.shape)  # (batch_size, seq_length, hidden_size)
print("RNN隐藏状态形状:", rnn_hidden.shape)  # (num_layers, batch_size, hidden_size)

# 2. LSTM (Long Short-Term Memory)
print("\n2. LSTM示例:")
lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=2,            # 使用2层LSTM
    batch_first=True,
    bidirectional=False      # 单向LSTM
)

# 前向传播
lstm_output, (lstm_hidden, lstm_cell) = lstm(input_data)
print("LSTM输出形状:", lstm_output.shape)
print("LSTM隐藏状态形状:", lstm_hidden.shape)
print("LSTM细胞状态形状:", lstm_cell.shape)

# 3. 双向LSTM
print("\n3. 双向LSTM示例:")
bilstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=True,
    bidirectional=True       # 双向LSTM
)

# 前向传播
bilstm_output, (bilstm_hidden, bilstm_cell) = bilstm(input_data)
print("双向LSTM输出形状:", bilstm_output.shape)  # hidden_size * 2 因为是双向
print("双向LSTM隐藏状态形状:", bilstm_hidden.shape)
print("双向LSTM细胞状态形状:", bilstm_cell.shape)

# 4. GRU (Gated Recurrent Unit)
print("\n4. GRU示例:")
gru = nn.GRU(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    batch_first=True
)

# 前向传播
gru_output, gru_hidden = gru(input_data)
print("GRU输出形状:", gru_output.shape)
print("GRU隐藏状态形状:", gru_hidden.shape)

# 5. 多层LSTM示例
print("\n5. 多层LSTM示例:")
multilayer_lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=3,            # 3层LSTM
    batch_first=True,
    dropout=0.2             # 层间添加dropout
)

# 前向传播
multilayer_output, (multilayer_hidden, multilayer_cell) = multilayer_lstm(input_data)
print("多层LSTM输出形状:", multilayer_output.shape)
print("多层LSTM隐藏状态形状:", multilayer_hidden.shape)
print("多层LSTM细胞状态形状:", multilayer_cell.shape)

# 6. 实际应用示例：序列分类
class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # 注意：因为是双向LSTM，所以hidden_size需要乘2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        # 分类层
        output = self.fc(last_hidden)
        return output

if __name__ == "__main__":
    print("\n序列分类示例:")
    # 创建模型实例
    num_classes = 3
    model = SequenceClassifier(input_size, hidden_size, num_classes)
    
    # 使用之前创建的输入数据
    output = model(input_data)
    print("分类输出形状:", output.shape)
    
    # 展示不同循环层的特点
    print("\n不同循环层的特点:")
    print("1. 简单RNN:")
    print("   - 最基本的循环结构")
    print("   - 可能存在梯度消失/爆炸问题")
    print("   - 适合简单的序列任务")
    
    print("\n2. LSTM:")
    print("   - 具有门控机制，可以处理长期依赖")
    print("   - 有细胞状态和隐藏状态")
    print("   - 适合复杂的序列任务")
    
    print("\n3. GRU:")
    print("   - LSTM的简化版本")
    print("   - 参数更少，训练更快")
    print("   - 性能通常与LSTM相当")
    
    print("\n4. 双向RNN:")
    print("   - 同时考虑过去和未来的信息")
    print("   - 输出维度是单向的两倍")
    print("   - 适合需要双向上下文的任务")
    
    # 简单的序列预测示例
    print("\n序列预测示例:")
    # 创建一个简单的时间序列数据
    time_steps = 10
    x = torch.linspace(0, 10, time_steps).reshape(1, -1, 1)
    y = torch.sin(x)  # 使用正弦函数生成目标值
    
    # 创建一个简单的LSTM预测模型
    class SimplePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.linear = nn.Linear(32, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.linear(lstm_out)
    
    predictor = SimplePredictor()
    pred_y = predictor(x)
    print("输入序列形状:", x.shape)
    print("预测输出形状:", pred_y.shape) 