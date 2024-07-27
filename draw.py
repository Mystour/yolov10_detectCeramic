import os
import cv2
import matplotlib.pyplot as plt
import random
import seaborn as sns

from ultralytics import YOLOv10
import pandas as pd

# 读取训练信息
df = pd.read_csv(r'.\runs\detect\train8\results.csv')
df.columns = df.columns.str.strip()

# 使用seaborn创建子图
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
# plot the columns using seaborn
# 训练过程中的边界框损失
sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0, 0])
# 训练过程中的分类损失
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0, 1])
# 训练过程中的距离损失
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1, 0])
# 训练过程中的精确度
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1, 1])
# 训练过程中的召回率
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2, 0])
# 训练过程中的平均精确度 (mAP) 在 50% IoU 阈值下的值，表示目标检测的平均准确度
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2, 1])
# 训练过程中的平均精确度 (mAP) 在 50% 到 95% IoU 阈值范围内的值，更综合地评估目标检测性能
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3, 0])
# 验证集中的边界框损失
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3, 1])
# 验证集中的分类损失
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4, 0])
# 验证集中的距离损失
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4, 1])
# set titles and axis labels for each subplot
axs[0, 0].set(title='Train Box Loss')
axs[0, 1].set(title='Train Class Loss')
axs[1, 0].set(title='Train DFL Loss')
axs[1, 1].set(title='Metrics Precision (B)')
axs[2, 0].set(title='Metrics Recall (B)')
axs[2, 1].set(title='Metrics mAP50 (B)')
axs[3, 0].set(title='Metrics mAP50-95 (B)')
axs[3, 1].set(title='Validation Box Loss')
axs[4, 0].set(title='Validation Class Loss')
axs[4, 1].set(title='Validation DFL Loss')
# 添加副标题和副标题
plt.suptitle('Training Metrics and Loss', fontsize=24)
# 调整上边距为副标题留出空间
plt.subplots_adjust(top=0.8)
# 调整子图之间的间距
plt.tight_layout()
plt.show()