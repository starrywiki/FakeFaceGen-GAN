# FaceFakeGen Report
## 代码解释
### Preprocess
预处理数据集
### Dataloader
- 自定义数据集类 ***DataGenerater*** 继承自torch.utils.data.Datas用于加载和预处理图像
    `__init__` 初始化数据集路径、文件列表、目标图像尺寸和变换
    `__getitem__` 加载单张图像，调整大小并转换格式，最终返回张量
    `__len__` 返回数据集的总样本数
- `train_loader` & `train_dataset` 创建数据集和数据加载器
### Generator 
生成对抗网络（GAN）的生成器部分,从随机噪声生成图像
- 加载模型：预训练的生成器模型。
- 输入噪声：随机噪声作为生成器的输入。
- 生成图像：模型将噪声映射为图像。
- 后处理：调整格式和数值范围。
- 保存结果：保存为PNG文件。
### Network（主要代码实现）
神经网络架构定义
两个核心网络：生成器&判别器
- ***Generator***:
    噪声向量 → 假图像
    输入: 100维随机噪声
    输出: 64×64×3 彩色人脸图像
- ***Discriminator***: 
    图像 → 真假判断
    输入: 64×64×3 图像
    输出: [0,1] 概率值（0=假，1=真）
### Train
超参数配置
```python
img_dim = 64          # 图像尺寸 64×64
lr = 0.0002          # 学习率
epochs = 5           # 训练轮数 
batch_size = 128     # 批量大小
G_DIMENSION = 100    # 生成器输入噪声维度
beta1 = 0.5         # Adam优化器参数1
beta2 = 0.999       # Adam优化器参数2
output_path = "output"  # 输出目录
real_label = 1      # 真实图像标签
fake_label = 0      # 假图像标签
```

网络实例化
```python
netD = Discriminator().to(device)  # 判别器
netG = Generator().to(device)      # 生成器
```

损失函数和优化器
```python
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
```

**BCELoss公式**: `Loss = -[y*log(x) + (1-y)*log(1-x)]`
- y=1时：Loss = -log(x) → 希望x接近1
- y=0时：Loss = -log(1-x) → 希望x接近0

---

主训练循环：Epoch级别循环
```python
for epoch in range(epochs):  # 5个epoch
    for batch_id, data in enumerate(tqdm(train_loader)):
        # 每个batch的训练逻辑
```


**判别器训练** 

1. 梯度清零
 ```python
    optimizerD.zero_grad()
```
**作用**: 清除上一次迭代的梯度累积

2. 真实图像训练
- 真实图像 → 判别器 → 概率值
- 期望输出: 接近1 (表示"真实")
- 损失: -log(判别器输出)

3. 假图像训练
```python
noise = torch.randn(current_batch_size, G_DIMENSION, 1, 1, device=device)  # 生成随机噪声
fake = netG(noise)                           # 生成器生成假图像
label.fill_(fake_label)                      # 标签设为0
output = netD(fake.detach()).view(-1)        # 判别器判断假图像
errD_fake = criterion(output, label)         # 计算假图像损失  
errD_fake.backward()                         # 反向传播
```

**关键点**:
- `fake.detach()`: 阻断梯度传播到生成器（只训练判别器）
- 期望输出: 接近0 (表示"假的")
- 损失: -log(1-判别器输出)

4. 更新判别器参数
```python
errD = errD_real + errD_fake                 # 总判别器损失
optimizerD.step()                            # 更新参数
```

---

**生成器训练** 

1. 梯度清零
```python
optimizerG.zero_grad()
```

2. 训练生成器"欺骗"判别器
```python
label.fill_(real_label)                      # 标签设为1 (期望被认为是真的)
output = netD(fake).view(-1)                 # 重新通过判别器 (不detach!)
errG = criterion(output, label)              # 生成器损失
errG.backward()                              # 反向传播
optimizerG.step()                            # 更新生成器参数
```

**核心思想**:
- 生成器的目标: 让判别器认为假图像是真的
- 期望判别器输出: 接近1
- 损失: -log(判别器对假图像的输出)

---

训练过程中的对抗机制

**判别器的视角**
1. **看到真实图像**: "这是真的，输出1"
2. **看到生成图像**: "这是假的，输出0" 
3. **训练目标**: 最大化分类准确性

**生成器的视角**
1. **生成图像**: 从噪声创造图像
2. **欺骗判别器**: 希望判别器输出1
3. **训练目标**: 最小化被识破的概率

**博弈过程**
```
初始状态: G生成很差 → D很容易识别 → D损失低，G损失高
训练进行: G逐渐改进 → D识别困难 → 两者损失趋于平衡
理想状态: D准确率≈50% → G生成逼真图像
```



**技术细节**

数据预处理
- 图像归一化到 [-1, 1] 范围
- 与生成器Tanh输出匹配

**标签平滑** 
```python
real_label = 0.9  # 而不是1.0
fake_label = 0.1  # 而不是0.0
```
**作用**: 提高训练稳定性，避免过度自信

### 梯度管理
- `detach()`: 控制梯度流向
- `zero_grad()`: 防止梯度累积
- `backward()`: 计算梯度
- `step()`: 更新参数

这整个过程就是让两个神经网络在"博弈"中相互改进，最终生成器学会创造逼真的人脸图像