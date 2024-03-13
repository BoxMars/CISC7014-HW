import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import pickle

from utils import get_label_names

cudnn.benchmark = True

class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 100)
        )


    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x
    
def test(model, testloader, criterion, batch_size=10, show_detail=False):
    classes = get_label_names()
    test_loss = 0.0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    for data, target in testloader:
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(np.squeeze(correct_tensor.cpu().numpy()))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(testloader.dataset)
    if show_detail:
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(100):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
    return np.sum(class_correct) / np.sum(class_total)

def training(model,trainloader,testloader,name):
    print("-"*50)
    print('Training Model {}'.format(name))
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    running_losses=[]
    accs=[]
    for epoch in range(400):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs=inputs.cuda()
            labels=labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        acc= test(model, testloader, criterion, show_detail=False)
        if acc>max(accs or [0]):
            torch.save(model.state_dict(), "best_model_{}.pth".format(name))
            print('Model Saved - {}'.format(epoch))
        accs.append(acc)
        print('Epoch: {}\tTraining Loss: {:.4f}\t Test Acc: {:.2f}%'.format(epoch,running_loss/i,acc*100))
        running_losses.append(running_loss/i)
    
    with open('loss-{}.pkl'.format(name), 'wb') as f:
        pickle.dump(running_losses, f)
    with open('accs-{}.pkl'.format(name), 'wb') as f:
        pickle.dump(accs, f)

    plt.plot(range(400), running_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Performance of Model {}".format(name))
    plt.savefig('performance-{}.png'.format(name))

    plt.plot(range(400), accs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Model {}".format(name))
    plt.savefig('acc-{}.png'.format(name))

    print('Finished Training')

    # load best model
    model.load_state_dict(torch.load("best_model_{}.pth".format(name)))
    test(model, testloader, criterion, show_detail=True)