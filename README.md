# ImagePaoGe
炮哥带你学
## LeNet5

### 网络架构
> LeNet5/net.py

## 输入输出
> 输入 torch.Size([1, 1, 28, 28])
>
> 输出 torch.Size([1, 10])

## 训练
```python
# 准备数据 
train_dataset.__len__       60000
train_dataloader.__len__    3750
test_dataset.__len__        10000
test_dataloader.__len__     625


for e in range(epoch):
    train(...)
    val(...)

    torch.save(...)

matplot_loss(...)
matplot_acc(...))

```

```text
'''
# 0 维度 10 个
# 1 维度 16 个

print(torch.max(output, 0)[0]) # 值
print(torch.max(output, 0)[1]) # 值
print(torch.max(output, 1)[0]) # 位置
print(torch.max(output, 1)[1]) # 位置
print("****************************************")
print(output.argmax(0))
print(output.argmax(1))
        '''

```