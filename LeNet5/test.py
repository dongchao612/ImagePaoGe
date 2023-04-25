import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage, ToTensor, Compose

from LeNet5.net import LeNet5

if __name__ == '__main__':
    # 数据转换为tensor格式
    data_transform = Compose([
        ToTensor()
    ])
    # 加载数据,train_dataloader是一个16*784的矩阵,一共3750个，test_dataloader是一个16*784的矩阵,一共625个
    train_dataset = MNIST(root="./data", train=True, transform=data_transform, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    test_dataset = MNIST(root="./data", train=False, transform=data_transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

    # 如果有显卡，可转到GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:{}".format(device))

    # 定义模型并转到GPU
    model = LeNet5().to(device)
    model.load_state_dict(torch.load("save_model/best_model.pth"))

    # 获取类别
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # 把tensor转换为图片方便可视化
    show = ToPILImage()
    num = test_dataset.__len__()
    acc_count = 0

    for i in range(num):
        img, target = test_dataset[i][0], test_dataset[i][1]
        # show(img).show()
        # 扩展张量维度为4维
        # torch.Size([1, 28, 28]) >> torch.Size([1, 1, 28, 28])
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).to(device)

        with torch.no_grad():
            pred = model(x)
            # 得到预测类别中最高的那一类，再把最高的这一类对应classes中的哪一类标签
            predicted, actual = classes[torch.argmax(pred, 1).item()], classes[target]

            # 最终输出的预测值与真实值
            # print(f'predicted: "{predicted}", actual:"{actual}"')
            if predicted == actual:
                acc_count = acc_count + 1

    print("acc:", acc_count / num * 100, "%")
