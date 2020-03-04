"""1"""
import os
from re import split as split
import shutil

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# 创建类别文件夹
if not os.path.exists("split/"):
    os.makedirs("split/")
for item in classification:
    os.mkdir("split/" + item)

imagePath = 'dataset/cifar-10-images'
imageFiles = []

for root, dirs, files in os.walk(imagePath):
    imageFiles += files

# 将训练集分类
print(imageFiles)
n = 0
for cls in classification:
    for i in range(len(imageFiles)):
        if split('[-.]', imageFiles[i])[2] == cls:
            shutil.copy(imagePath + '/' + imageFiles[i], 'split\\' + cls)
            n += 1
            print(n)
