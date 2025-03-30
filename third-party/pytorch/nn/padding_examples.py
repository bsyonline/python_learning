import torch
import torch.nn as nn


input_tensor = torch.Tensor([[[1,2],[3,4]]])
print("原始输入张量形状:", input_tensor.shape)
print("原始输入张量:\n", input_tensor)

# 1. ZeroPad2d - 在2D输入的四周添加零填充
zero_pad = nn.ZeroPad2d(padding=(1, 1, 2, 2))  # 左1, 右1, 上2, 下2
zero_padded = zero_pad(input_tensor)
print("\nZeroPad2d 后的形状:", zero_padded.shape)
print("zero_padded:\n", zero_padded)
print("填充说明: 左右各填充1个零，上下各填充2个零")

# 2. ReplicationPad2d - 复制边界值进行填充
replication_pad = nn.ReplicationPad2d(padding=2)  # 四周都填充2个像素
replicated = replication_pad(input_tensor)
print("\nReplicationPad2d 后的形状:", replicated.shape)
print("replicated:\n", replicated)
print("填充说明: 四周都用边界值填充2个像素")

# 3. ReflectionPad2d - 镜像填充
reflection_pad = nn.ReflectionPad2d(padding=1)  # 四周都填充1个像素
reflected = reflection_pad(input_tensor)
print("\nReflectionPad2d 后的形状:", reflected.shape)
print("reflected:\n", reflected)
print("填充说明: 四周都进行镜像填充2个像素")

# 4. ConstantPad2d - 使用常数值填充
constant_pad = nn.ConstantPad2d(padding=1, value=0.5)  # 四周填充2个像素，填充值为0.5
constant_padded = constant_pad(input_tensor)
print("\nConstantPad2d 后的形状:", constant_padded.shape)
print("constant_padded:\n", constant_padded)
print("填充说明: 四周都用0.5填充2个像素")

# 5. 在卷积层中使用padding参数
conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
conv_output = conv2d(input_tensor)
print("\n使用padding的卷积层输出形状:", conv_output.shape)
print("conv_output:\n", conv_output)
print("填充说明: 卷积层中padding=1保持输入输出大小相同")

if __name__ == "__main__":
    # 运行一个具体的例子
    print("\n详细的填充示例:")
    small_input = torch.tensor([[1, 2], [3, 4]]).float().unsqueeze(0).unsqueeze(0)
    print("小型输入张量:\n", small_input.squeeze())
    
    # 使用ReplicationPad2d
    rep_pad = nn.ReplicationPad2d(1)
    padded = rep_pad(small_input)
    print("\n使用ReplicationPad2d填充后:\n", padded.squeeze()) 