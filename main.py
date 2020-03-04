"""6"""
from train_func import *
from alexNet import *
import torch
import torch.utils.data as Data
from torchvision import datasets, transforms

NET = NoBnNet
EPOCH = 30
BATCH_SIZE = 100
LEARNING_RATE = 0.001
trainPath = 'train'
batchAccList = []

if __name__ == '__main__':
    dataTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    imageDataset = datasets.ImageFolder(trainPath, dataTransforms)

    dataLoader = Data.DataLoader(dataset=imageDataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    myCNN = NET().cuda()

    optimizer = torch.optim.Adam(myCNN.parameters(), lr=LEARNING_RATE)
    lossFunc = nn.CrossEntropyLoss().cuda()

    trainedModel = train_func(myCNN, EPOCH, dataLoader, lossFunc, optimizer)

    if NET == BnNet:
        torch.save(trainedModel, 'bn_net.pkl')
    else:
        torch.save(trainedModel, 'no_bn_net.pkl')
