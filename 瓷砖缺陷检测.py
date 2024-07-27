import os
import cv2
import matplotlib.pyplot as plt
import random
import seaborn as sns

from ultralytics import YOLOv10
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sns.set_style('darkgrid')

train_images = r'.\Ceramic tile surface defect detection system and application\tile_data\images'
train_labels = r'.\Ceramic tile surface defect detection system and application\tile_data\labels'
val_images = r'.\Ceramic tile surface defect detection system and application\val_data\images'
val_labels = r'.\Ceramic tile surface defect detection system and application\val_data\labels'
# 获取指定文件夹路径 train_images 下的所有文件和子文件夹的名称，并将它们存储在一个列表 image_files 中。
image_files = os.listdir(train_images)
# 从列表中随机选择16个镜像文件
random_images = random.sample(image_files, 15)
# 设置画布
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for i, image_file in enumerate(random_images):
    row = i // 4
    col = i % 4
    # 加载图像
    # 通过os.path.join()函数，将文件夹路径和文件名合并成一个完整的路径。
    image_path = os.path.join(train_images, image_file)
    # 通过将image_path作为参数传递给imread()函数，它将读取该路径下的图像文件，并将其加载为一个图像对象，存储在变量image中。
    image = cv2.imread(image_path)

    # 为这个数据的图像加载标签数据
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(train_labels, label_file)
    with open(label_path, "r") as f:
        labels = f.read().strip().split("\n")
    # 循环遍历标签并绘制目标检测
    # 通过循环处理每个标签，并根据标签信息计算出相应的矩形框坐标，可以在图像上绘制出所有目标框，以可视化检测结果。
    for label in labels:
        if len(label.split()) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, label.split())
        x_min = int((x_center - width/2) * image.shape[1])
        y_min = int((y_center - height/2) * image.shape[0])
        x_max = int((x_center + width/2) * image.shape[1])
        y_max = int((y_center + height/2) * image.shape[0])
        # 绘制一个矩形框。
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    # 显示带有目标检测的图像
    axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[row, col].axis('off')

# 从训练图像文件夹中获取所有图像文件的文件名列表
image_files = os.listdir(train_images)
# 随机选择一个图像文件
random_images = random.sample(image_files, 1)
# 遍历随机选择的图像文件列表
for i, image_file in enumerate(random_images):
    # 构建图像文件的完整路径
    image_path = os.path.join(train_images, image_file)
    # 使用 OpenCV 加载图像
    image = cv2.imread(image_path)
    # 获取加载图像的尺寸信息
    height, width, channels = image.shape
    # 打印图像尺寸信息
    print(f"The image has dimensions {width}x{height} and {channels} channels.")
    break
    
# 从预训练模型加载模型
# model = YOLO('yolov5s.pt')

model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# You can also try YOLOv8 by switching line above to line bellow. I ran it first. It shows similar results.
# model = YOLO('yolov8n.pt')
# Training the model
model.train(
    data='./Ceramic tile surface defect detection system and application/tile.yaml',
            epochs=100,
            imgsz=(height, width, channels),# 输入图像的尺寸，这个值在之前的代码中已经获取到了。
            seed=42,# 随机种子，用于控制随机性。
            batch=1,# 训练的批次大小。
            workers=4, # 用于数据加载的工作线程数。
)
# 读取训练信息

df = pd.read_csv(r'.\runs\detect\train6\results.csv')
df.columns = df.columns.str.strip()

# 使用seaborn创建子图
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
# plot the columns using seaborn
# 训练过程中的边界框损失
sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
# 训练过程中的分类损失
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
# 训练过程中的距离损失
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
# 训练过程中的精确度
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
# 训练过程中的召回率
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
# 训练过程中的平均精确度 (mAP) 在 50% IoU 阈值下的值，表示目标检测的平均准确度
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
# 训练过程中的平均精确度 (mAP) 在 50% 到 95% IoU 阈值范围内的值，更综合地评估目标检测性能
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
# 验证集中的边界框损失
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
# 验证集中的分类损失
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
# 验证集中的距离损失
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])
# set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')
# 添加副标题和副标题
plt.suptitle('Training Metrics and Loss', fontsize=24)
# 调整上边距为副标题留出空间
plt.subplots_adjust(top=0.8)
# 调整子图之间的间距
plt.tight_layout()
plt.show()
