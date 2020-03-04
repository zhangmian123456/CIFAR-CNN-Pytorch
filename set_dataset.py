"""3"""
import re
import os
from skimage import io
from torchvision import datasets

splitPath = 'split_text.txt'

trainDataset = datasets.ImageFolder('split')

readLine = open(splitPath)
ttLabels = readLine.readlines()
i = 0

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# 在test和train文件夹中创建类别文件夹
for item in classification:
    os.mkdir("test/" + item)
    os.mkdir("train/" + item)

for j in range(len(trainDataset.imgs)):  # len(trainDataset.imgs)
    if re.split('[ \n]', ttLabels[j])[1] == '0':
        path = trainDataset.imgs[j][0]
        print(path)
        img = io.imread(path)
        folder = path.split('\\')[-2]
        imgName = path.split('\\')[-1]

        io.imsave('test/' + folder + '/' + imgName, img)
    else:
        path = trainDataset.imgs[j][0]
        print(path)
        img = io.imread(path)
        folder = path.split('\\')[-2]
        imgName = path.split('\\')[-1]

        io.imsave('train/' + folder + '/' + imgName, img)
