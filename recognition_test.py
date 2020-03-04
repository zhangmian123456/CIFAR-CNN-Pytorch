"""7"""
import numpy as np
import torch
import os
from PIL import Image
from alexNet import *
import torch.utils.data as Data
from torchvision import datasets, transforms
from main import NET

imagePath = 'test_image/'
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']


def image_process(img_path):
    im = Image.open(img_path)
    im_array = np.array(im)
    im_array = np.transpose(im_array, [2, 0, 1])
    return im_array


def image_recognition(image_path):
    image_files = []
    image_data = []
    pred_list = []
    for root, dirs, files in os.walk(image_path):
        image_files += files

    for image in image_files:
        image_data.append(image_process(image_path + image))

    print(image_files)
    test_data = np.stack(image_data, axis=0)

    x = torch.tensor(test_data, dtype=torch.float)

    trained_net = torch.load('net.pkl')

    prediction = trained_net(x.cuda()).cuda()
    preds = torch.max(prediction.data, 1)[1]

    for pred in preds:
        pred_list.append(classification[int(pred)])

    print(preds)
    print(pred_list)


def dataset_test_func(batch_size):
    accuracy_sum = 0
    running_corrects = 0
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_dataset = datasets.ImageFolder('test', data_transforms)

    data_loader = Data.DataLoader(dataset=image_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    dataset_size = len(data_loader.dataset)
    if NET == BnNet:
        trained_net = torch.load('bn_net.pkl').cuda()

    else:
        trained_net = torch.load('no_bn_net.pkl').cuda()

    for data in data_loader:
        inputs, labels = data

        outputs = trained_net(inputs.cuda())
        preds = torch.max(outputs.data, 1)[1]
        running_corrects += torch.sum(preds.cpu().data == labels.data).item()
        acc = running_corrects / 1000
        accuracy_sum += acc
        running_corrects = 0
    test_acc = accuracy_sum * batch_size / dataset_size
    return test_acc

# image_recognition(imagePath)


testResult = dataset_test_func(1000)
print(testResult)
