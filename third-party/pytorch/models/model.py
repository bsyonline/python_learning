import torch
from torch import nn
from torch.nn import Conv2d, Sequential, Flatten


class MyNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    myNN = MyNN()
    input = torch.ones((64, 3, 32, 32))
    output = myNN(input)
    print(output.shape)