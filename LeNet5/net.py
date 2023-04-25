import torch
from torch.nn import Module, Conv2d, Sigmoid, AvgPool2d,Flatten,Linear


# 定义一个网络模型类
class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        self.Sigmoid = Sigmoid()

        self.s2 = AvgPool2d(kernel_size=2, stride=2)
        self.c3 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = AvgPool2d(kernel_size=2, stride=2)
        self.c5 = Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.flatten = Flatten()

        self.f6 = Linear(120, 84)
        self.output = Linear(84, 10)

    def forward(self, x):

        x = self.Sigmoid(self.c1(x))

        x = self.s2(x)

        x = self.Sigmoid(self.c3(x))

        x = self.c5(self.s4(x))

        x = self.flatten(x)

        x = self.output(self.f6(x))

        return x


if __name__ == '__main__':
    bs, cin_channel, cin_w, cin_h = 1, 1, 28, 28
    x = torch.randn(bs, cin_channel, cin_w, cin_h)

    net = LeNet5()

    print(x.shape)
    print(net(x).shape)
