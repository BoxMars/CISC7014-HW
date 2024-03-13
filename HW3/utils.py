import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import sklearn.metrics
import torch.nn.functional as F

CURRENT_DIR = os.getcwd()
CIFAR_DIR = os.path.join(CURRENT_DIR, 'data','cifar')

def get_label_names():
    with open(os.path.join(CIFAR_DIR, 'cifar-100-python', 'meta'), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
    labelnames = labelnames[b'fine_label_names']
    for i in range(len(labelnames)):
        labelnames[i] = labelnames[i].decode("utf-8") 
    return labelnames


def plot_cifar_example():
    labelnames=get_label_names()

    with open(os.path.join(CIFAR_DIR, 'cifar-100-python', 'test'), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'fine_labels']
    fig, axes1 = plt.subplots(10,10,figsize=(12,16))

    i=0
    for j in range(10):
        for k in range(10):
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(imgList[i].transpose(1,2,0))
            axes1[j][k].set_title(labelnames[labelList[i]])
            i+=1
    plt.savefig('cifar100_example.png')

def plot_per_class_accuracy(models_dict, dataloaders, labelnames, img_num_per_cls, name,nClasses=100, device = 'cuda'):
    result_dict = {}
    for label in models_dict:
        model = models_dict[label]
        acc_per_class = get_per_class_acc(model, dataloaders, nClasses= nClasses, device= device)
        result_dict[label] = acc_per_class

    plt.figure(figsize=(15,4), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(100)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    plt.title('per-class accuracy vs. per-class #images - {}'.format(name), fontsize=20)
    ax1 = plt.gca()    
    ax2=ax1.twinx()
    for label in result_dict:
        ax1.bar(list(range(100)), result_dict[label], alpha=0.7, width=1, label= label, edgecolor = "black")
        
    ax1.set_ylabel('accuracy', fontsize=16, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

    ax2.set_ylabel('#images', fontsize=16, color='r')
    # print(img_num_per_cls)
    ax2.plot(img_num_per_cls, linewidth=4, color='r')
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)
    
    ax1.legend(prop={'size': 14})

    plt.savefig('img/pre-class-accuracy-vs-per-class-#images-{}.png'.format(name))
    print("img/pre-class-accuracy-vs-per-class-#images-{}.png is saved in the current directory".format(name))

def get_per_class_acc(model, dataloaders, nClasses= 100, device = 'cuda'):
    predList = np.array([])
    grndList = np.array([])
    model.eval()
    for sample in dataloaders:
        with torch.no_grad():
            images, labels = sample
            # print(images.shape)
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)   

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))    
            grndList = np.concatenate((grndList, labels))


    confMat = sklearn.metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1,1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i,i]

    acc_avgClass /= confMat.shape[0]
    
    acc_per_class = [0] * nClasses

    for i in range(nClasses):
        acc_per_class[i] = confMat[i,i]
    
    return acc_per_class 


if __name__ == "__main__":
    plot_cifar_example()
    print("cifar100_example.png is saved in the current directory")
    print("Please check the current directory for the image file")

