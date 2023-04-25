import torch
from torchvision.datasets import ImageFolder
from AlexNet.net import AlexNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

if __name__ == '__main__':
    tain_root = r"./data/train"
    test_root = r"./data/test"

    # 数据处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 重置大小到224*224
        transforms.RandomVerticalFlip(),  # 按照概率p对PIL图片进行垂直翻转，数据增强
        transforms.ToTensor(),  # 变成Tensor格式
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 将图像像素归一化到[-1,1]之间
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 将图像像素归一化到[-1,1]之间
    ])

    train_data = ImageFolder(root=tain_root, transform=train_transform)
    test_data = ImageFolder(root=test_root, transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)  # 训练集
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)  # 测试集合

    # 如果有显卡，可转到GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:{}".format(device))

    # 定义模型并转到GPU
    model = AlexNet().to(device)
    model.load_state_dict(torch.load("save_model/best_model.pth"))

    # 获取类别
    classes = ["ants", "bees"]
    # 把tensor转换为图片方便可视化

    show = ToPILImage()
    num = test_data.__len__()
    acc_count = 0

    for i in range(num):
        img, target = test_data[i][0], test_data[i][1]
        # show(img).show()
        # 扩展张量维度为4维
        # torch.Size([3, 224, 224]) >> torch.Size([1, 3, 224, 224])
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).to(device)

        with torch.no_grad():
            pred = model(x)
            # print(pred)
            # 得到预测类别中最高的那一类，再把最高的这一类对应classes中的哪一类标签
            predicted, actual = classes[torch.argmax(pred, 1).item()], classes[target]

            # 最终输出的预测值与真实值
            # print(f'predicted: "{predicted}", actual:"{actual}"')
            if predicted == actual:
                acc_count = acc_count + 1

    print("acc:", acc_count / num * 100, "%")
