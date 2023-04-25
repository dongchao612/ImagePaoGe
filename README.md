# ImagePaoGe
炮哥带你学

## LeNet5

### 网络架构
> LeNet5/net.py

### 输入输出
> 输入 torch.Size([1, 1, 28, 28])
>
> 输出 torch.Size([1, 10])

### 训练

- 数据集 MNIST

  - batchSize           32
  - train_dataset       60000
  - train_dataloader    3750
  - test_dataset        10000
  - test_dataloader     625

- 损失函数  CrossEntropyLoss
- 优化器    SGD
- epoch    100
- 精度      99.61


## AlexNet

### 网络架构
> AlexNet/net.py

### 输入输出
> 输入 torch.Size([1, 3, 224, 224])
>
> 输出 torch.Size([1, 2])

### 训练

- 数据集 蚂蚁-蜜蜂

  - batchSize           32
  - train_dataset       244
  - train_dataloader    3750
  - test_dataset        8
  - test_dataloader     5

- 损失函数  CrossEntropyLoss
- 优化器    SGD
- epoch     100
- 精度      62.09
