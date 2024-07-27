import os
import cv2
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd

# 读取训练信息
df = pd.read_csv(r'.\runs\detect\train8\results.csv')
df.columns = df.columns.str.strip()

# 使用seaborn创建子图
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))

# 训练过程中的边界框损失
sns.lineplot(x='epoch', y='train/box_om', data=df, ax=axs[0, 0])
# 训练过程中的分类损失
sns.lineplot(x='epoch', y='train/cls_om', data=df, ax=axs[0, 1])
# 训练过程中的距离损失
sns.lineplot(x='epoch', y='train/dfl_om', data=df, ax=axs[1, 0])
# 验证集中的边界框损失
sns.lineplot(x='epoch', y='val/box_om', data=df, ax=axs[1, 1])
# 验证集中的分类损失
sns.lineplot(x='epoch', y='val/cls_om', data=df, ax=axs[2, 0])
# 验证集中的距离损失
sns.lineplot(x='epoch', y='val/dfl_om', data=df, ax=axs[2, 1])

# 设置每个子图的标题和轴标签
axs[0, 0].set(title='Train Box Loss')
axs[0, 1].set(title='Train Class Loss')
axs[1, 0].set(title='Train DFL Loss')
axs[1, 1].set(title='Validation Box Loss')
axs[2, 0].set(title='Validation Class Loss')
axs[2, 1].set(title='Validation DFL Loss')

# 添加副标题和副标题
plt.suptitle('Training Metrics and Loss', fontsize=24)
# 调整上边距为副标题留出空间
plt.subplots_adjust(top=0.8)
# 调整子图之间的间距
plt.tight_layout()
plt.show()