"""训练和测试同时进行"""
from model import *
import torch
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


EPOCH = 20
BATCH_SIZE = 100
LEARNING_RATE = 0.001
batchAccList = []

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(x,
                            data_transforms[x])
    for x in ['train', 'test']
}

dataLoaders = {
    x: Data.DataLoader(dataset=image_datasets[x],
                       batch_size=BATCH_SIZE,
                       shuffle=True)
    for x in ['train', 'test']
}

datasets_sizes = {
    x: len(image_datasets[x])
    for x in ['train', 'test']
}


myCNN = CNNNet().cuda()
optimizer = torch.optim.Adam(myCNN.parameters(), lr=LEARNING_RATE)
lossFunc = nn.CrossEntropyLoss().cuda()

# best_acc = 0.0

for epoch in range(EPOCH):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + 1, EPOCH))

    for phase in ['train', 'test']:
        if phase == 'train':
            myCNN.train(True)
        else:
            myCNN.train(False)

        running_loss = 0.0
        running_corrects = 0
        batch_train = 10
        for data in dataLoaders[phase]:
            '''data即为一个BATCH的训练集'''
            '''一共 dataset/batch 个 data'''
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = myCNN(inputs)

            preds = torch.max(outputs.data, 1)[1]
            loss = lossFunc(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            if phase == 'train':
                '''记录训练准确率变化'''
                batch_train += 1
                if batch_train % 10 == 0:
                    batch_acc = running_corrects / (batch_train * BATCH_SIZE)
                    batchAccList.append(batch_acc)

        epoch_loss = running_loss / datasets_sizes[phase]
        epoch_acc = running_corrects / datasets_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc
        ))
        # 记录最佳参数
        # if phase == 'test' and epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     best_model_wts = myCNN.state_dict()

plt.plot(batchAccList)
plt.show()
