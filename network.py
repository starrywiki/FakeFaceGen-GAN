# neteork.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    判别器网络：判断输入图像是真实的还是生成的
    输入:64x64x3的图像
    输出:单个概率值(0-1之间)
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # 第一层：64x64x3 -> 32x32x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)

        # 第二层：32x32x64 -> 16x16x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # 第三层：16x16x128 -> 8x8x256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # 第四层：8x8x256 -> 4x4x512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        # 最后一层：4x4x512 -> 1x1x1
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

        # LeakyReLU激活函数，斜率为0.2
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Sigmoid激活函数用于最终输出
        self.sigmoid = nn.Sigmoid()

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化:零中心正态分布:标准差0.02"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # 输入x的形状应该是 [batch_size, 3, 64, 64]

        # 第一层
        x = self.conv1(x)
        x = self.leaky_relu(x)

        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        # 第四层
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        # 最后一层
        x = self.conv5(x)
        x = self.sigmoid(x)

        # 展平为 [batch_size, 1]
        return x.view(-1, 1)


class Generator(nn.Module):
    """
    生成器网络:从随机噪声生成64x64x3的图像
    输入:100维随机噪声向量
    输出:64x64x3的图像
    """

    def __init__(self, nz=100):
        super(Generator, self).__init__()
        self.nz = nz

        # 第一层：100x1x1 -> 512x4x4
        self.convt1 = nn.ConvTranspose2d(
            nz, 512, kernel_size=4, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(512)

        # 第二层：512x4x4 -> 256x8x8
        self.convt2 = nn.ConvTranspose2d(
            512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(256)

        # 第三层：256x8x8 -> 128x16x16
        self.convt3 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)

        # 第四层：128x16x16 -> 64x32x32
        self.convt4 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(64)

        # 最后一层：64x32x32 -> 3x64x64
        self.convt5 = nn.ConvTranspose2d(
            64, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化:零中心正态分布:标准差0.02"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):

        # 第一层
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 第二层
        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 第三层
        x = self.convt3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # 第四层
        x = self.convt4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # 最后一层
        x = self.convt5(x)
        x = self.tanh(x)

        return x
