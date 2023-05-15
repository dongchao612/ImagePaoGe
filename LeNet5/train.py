import os

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from LeNet5.net import LeNet5
import matplotlib

matplotlib.use("qt5agg")

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

torch.backends.enabled = True
torch.backends.benchmark = True


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 将模型转换为训练模式
    model.train()
    loss, current, n = 0.0, 0.0, 0.0
    for batch, (x, y) in enumerate(dataloader):  # 一次是16个数据，一共循环3750次
        # 前向传播
        x, y = x.to(device), y.to(device)

        output = model(x)

        cur_acc = sum(y == output.argmax(1)) / output.shape[0]  # tensor(0.1875, device='cuda:0')

        cur_loss = loss_fn(output, y)  # 损失函数  cur_loss.item()  2.263385772705078

        # 梯度更新
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        # 计算loss
        loss += cur_loss.item()
        current += cur_acc.item()

        n += 1

    # 计算总的loss平均值,一共3750个的平均值
    # print("train_loss:", loss / n)
    # print("train_acc:", (current / n) * 100)

    return loss / n, current / n


def val(dataloader, model, loss_fn):
    # 将模型转换为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)

            cur_acc = sum(output.argmax(1) == y) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    # print("val_loss:", loss / n)
    # print("val_acc:", (current / n) * 100)

    return loss / n, current / n


# 定义画图函数
def matplot_loss(train_loss_list, val_loss_list):
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.legend(loc="best")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("训练集和测试集loss对比图")
    plt.savefig("loss")
    plt.show()


def matplot_acc(train_acc_list, val_acc_list):
    plt.plot(train_acc_list, label="train_acc")
    plt.plot(val_acc_list, label="val_acc")
    plt.legend(loc="best")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.title("训练集和测试集acc对比图")
    plt.ylim(0, 1)
    plt.savefig("accuracy")
    plt.show()


if __name__ == '__main__':
    # 数据转换为tensor格式
    data_transform = transforms.Compose([
        ToTensor()
    ])
    batchSize = 32

    # 加载数据,train_dataloader是一个16*784的矩阵,一共3750个，test_dataloader是一个16*784的矩阵,一共625个
    train_dataset = MNIST(root="./data", train=True, transform=data_transform, download=True)
    test_dataset = MNIST(root="./data", train=False, transform=data_transform, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, drop_last=True)

    print("train_dataset.__len__():", train_dataset.__len__())  # 60000
    print("train_dataloader.__len__():", train_dataloader.__len__())  # 3750
    print("test_dataset.__len__():", test_dataset.__len__())  # 10000
    print("test_dataloader.__len__():", test_dataloader.__len__())  # 625

    # 如果有显卡，可转到GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:{}".format(device))

    # 定义模型并转到GPU
    model = LeNet5().to(device)

    # 定义一个损失函数:交叉熵函数
    loss_fn = CrossEntropyLoss().to(device)

    # 定义一个优化器
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 学习率每隔10轮变为原来的0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # train(train_dataloader, model, loss_fn, optimizer)
    # val(test_dataloader, model, loss_fn)

    # 开始训练
    epoch = 100
    min_acc = 0

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for e in range(epoch):
        print("================= EPOCH: {}/{} =================".format(e + 1, epoch))

        # 训练函数
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 验证函数
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
            print('save best model', "train_acc=", train_acc, "val_acc=", val_acc)
            torch.save(model.state_dict(), f"{folder}/best_model.pth")

    print("训练完成,开始画图...")
    matplot_loss(train_loss_list=train_loss_list, val_loss_list=val_loss_list)
    matplot_acc(train_acc_list=train_acc_list, val_acc_list=val_acc_list)
