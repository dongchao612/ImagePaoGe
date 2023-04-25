import torch
from torch.nn import Module, Conv2d, Sigmoid, AvgPool2d, Flatten, Linear, ReLU, MaxPool2d, Dropout
from torch import nn


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=(4, 4), padding=2)
        self.r1 = ReLU(inplace=True)
        self.m1 = MaxPool2d(2)

        self.c2 = Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.r2 = ReLU(inplace=True)
        self.m2 = MaxPool2d(2)

        self.c3 = Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.r3 = ReLU(inplace=True)

        self.c4 = Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.r4 = ReLU(inplace=True)

        self.c5 = Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.r5 = ReLU(inplace=True)

        self.m3 = MaxPool2d(kernel_size=3, stride=2)

        self.flatten = Flatten()

        self.f6 = Linear(4608, 2048)
        self.d1 = Dropout(p=0.5)
        self.f7 = Linear(2048, 2048)
        self.d2 = Dropout(p=0.5)
        self.f8 = Linear(2048, 1000)
        self.d3 = Dropout(p=0.5)
        self.f9 = Linear(1000, 2)

    def forward(self, x):

        x = self.m1(self.r1(self.c1(x)))

        x = self.m2(self.r2(self.c2(x)))
        x = self.r3(self.c3(x))

        x = self.r4(self.c4(x))

        x = self.r1(self.c5(x))

        x = self.m3(x)

        x = self.flatten(x)

        x = self.d1(self.f6(x))

        x = self.d2(self.f7(x))

        x = self.d3(self.f8(x))

        x = self.f9(x)

        return x


if __name__ == "__main__":
    bs, cin_channel, cin_w, cin_h = 1, 3, 224, 224
    x = torch.randn(bs, cin_channel, cin_w, cin_h)

    net = AlexNet()
    print(x.shape)
    print(net(x).shape)
