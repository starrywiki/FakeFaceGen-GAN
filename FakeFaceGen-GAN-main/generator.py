import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from network import Generator
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义生成器
generate = Generator().to(device)

# 加载模型权重
state_dict = torch.load("generator.params", map_location=device)
generate.load_state_dict(state_dict)

# 生成噪声
noise = torch.randn(100, 100, 1, 1, dtype=torch.float32).to(device)

# 生成图像
with torch.no_grad():
    generated_image = generate(noise).cpu().numpy()

os.makedirs('Generate', exist_ok=True)  # 创建目录以保存生成的图像

# 保存生成的图像
for j in range(100):
    image = generated_image[j].transpose(1, 2, 0)  # 转换为 HWC 格式
    image = ((image + 1) / 2 * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    pil_image.save(f'Generate/generated_{j + 1}.png')

print("Images saved using Pillow (PIL)")