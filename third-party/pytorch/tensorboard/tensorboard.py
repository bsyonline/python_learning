from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(".")

for i in range(100):
    writer.add_scalar("train", i * i, i)

writer.close()