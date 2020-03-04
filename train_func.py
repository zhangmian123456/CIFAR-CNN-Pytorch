"""5"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


def train_func(model, epochs, data_loader, loss_func, optimizer):
    dataset_size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    batch_acc_list = []
    batch_record = 1
    train_times = (dataset_size * epochs) / batch_size
    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        running_loss = 0.0
        running_corrects = 0
        batch_train = 0

        for data in data_loader:
            '''data即为一个BATCH的训练集'''
            '''一共 dataset/batch 个 data'''
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(inputs)

            preds = torch.max(outputs.data, 1)[1]
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

            batch_train += 1
            if batch_train % batch_record == 0:
                batch_acc = running_corrects / (batch_train * batch_size)
                batch_acc_list.append(batch_acc)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc
        ))

    plt.plot(np.linspace(0, train_times, int(train_times/batch_record)), batch_acc_list)
    plt.show()
    return model
