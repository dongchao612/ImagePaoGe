import os

import matplotlib.pylab as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from AlexNet.net import AlexNet

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


torch.backends.enabled = True
torch.backends.benchmark = True


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 将模型转换为训练模式
    model.train()
    loss, current, n = 0.0, 0.0, 0.0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)

        output = model(image)
        # print(image.shape, y.shape,output.shape)
        # torch.Size([32, 2]) torch.Size([32])
        curr_loss = loss_fn(output, y)
        cur_acc = sum(y == output.argmax(1)) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

        # 计算loss
        loss += curr_loss.item()
        current += cur_acc.item()
        n = n + 1

    #print("train_loss:", loss / n)
    #print("train_acc:", (current / n) * 100)

    return loss / n, current / n


# 定义一个验证函数
def val(dataloader, model, loss_fn):
    # 将模型转换为验证验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            curr_loss = loss_fn(output, y)
            cur_acc = sum(y == output.argmax(1)) / output.shape[0]

            loss += curr_loss.item()
            current += cur_acc.item()
            n = n + 1

    #print("val_loss:", loss / n)
    #print("val_acc:", (current / n) * 100)

    return loss / n, current / n


# 定义画图函数
def matplot_loss(train_loss_list, val_loss_list):
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.legend(loc="best")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("loss")
    plt.title("训练集和测试集loss对比图")
    plt.show()


def matplot_acc(train_acc_list, val_acc_list):
    plt.plot(train_acc_list, label="train_acc")
    plt.plot(val_acc_list, label="val_acc")
    plt.legend(loc="best")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.title("训练集和测试集acc对比图")
    plt.savefig("accuracy")
    plt.show()


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

    batchSize = 32
    train_data = ImageFolder(root=tain_root, transform=train_transform)
    test_data = ImageFolder(root=test_root, transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=batchSize, shuffle=True, drop_last=True)  # 训练集
    test_dataloader = DataLoader(test_data, batch_size=batchSize, shuffle=True, drop_last=True)  # 测试集合

    print("train_data.__len__():", train_data.__len__())  # 244
    print("test_data.__len__():", test_data.__len__())  # 153
    print("train_dataloader.__len__():", train_dataloader.__len__())  # 8
    print("test_dataloader.__len__():", test_dataloader.__len__())  # 5

    # 如果有显卡，可转到GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:{}".format(device))

    # 定义模型并转到GPU
    model = AlexNet().to(device)

    # 定义一个损失函数
    loss_fn = CrossEntropyLoss().to(device)

    # 定义一个优化器
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 学习率每隔10轮变为原来的0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # train(train_dataloader, model, loss_fn, optimizer)  # train_loss: 0.6938427910208702 train_acc: 47.42187522351742
    # val(test_dataloader, model, loss_fn)#val_loss: 0.6920500755310058 val_acc: 56.99999928474426

    # 开始训练
    epoch = 100
    min_acc = 0

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for e in range(epoch):
        print("================= EPOCH: {}/{} ===============".format(e + 1, epoch))

        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss, val_acc = val(test_dataloader, model, loss_fn)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        scheduler.step()


        # 保存最好的模型权重
        if val_acc > min_acc:
            folder = "save_model"
            if not os.path.exists(folder):
                os.mkdir(folder)
            min_acc = val_acc
            print('save best model',"train_acc=",train_acc,"val_acc=",val_acc)
            torch.save(model.state_dict(), f"{folder}/best_model.pth")


    #print("训练完成,开始画图...")
    matplot_loss(train_loss_list=train_loss_list, val_loss_list=val_loss_list)
    matplot_acc(train_acc_list=train_acc_list, val_acc_list=val_acc_list)


