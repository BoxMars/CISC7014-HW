import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_per_class_accuracy, get_label_names
import data
from model import CNN
from pytorch_cifar.models.resnet import ResNet, BasicBlock, Bottleneck
from pytorch_cifar.models.mobilenetv2 import MobileNetV2

models=[
    'CNN',
    'ResNet18',
    'ResNet50',
    'ResNet152',
    'MobileNetV2',
]

color=[
    'b',
    'g',
    'r',
    'c',
    'm',
]
train_dataset = data.get_CIFAR100LT_data('train')

test_dataset = data.get_CIFAR100LT_data('test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

def load_loss(name):
    with open(f'loss-{name}.pkl', 'rb') as f:
        return pickle.load(f)

def load_acc(name):
    with open(f'accs-{name}.pkl', 'rb') as f:
        return pickle.load(f)

def load_best_models(name):
    if name == 'CNN':
        model = CNN()
        model.load_state_dict(torch.load(f'best_model_{name}.pth'))
    elif name == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2],100)
        model.load_state_dict(torch.load(f'best_model_{name}.pth'))
    elif name == 'ResNet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3],100)
        model.load_state_dict(torch.load(f'best_model_{name}.pth'))
    elif name == 'ResNet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3],100)
        model.load_state_dict(torch.load(f'best_model_{name}.pth'))
    elif name == 'MobileNetV2':
        model = MobileNetV2(num_classes=100)
        model.load_state_dict(torch.load(f'best_model_{name}.pth'))
    model.to('cuda')
    return model

if __name__ == "__main__":
    for i,model in enumerate(models):
        plt.plot(range(400), load_loss(model))
        print(load_loss(model))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Performance of Model {model}")
        plt.savefig(f'img/performance_{model}.png')
        plt.clf()

        plt.plot(range(400), load_acc(model))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy of Model {model}")
        plt.savefig(f'img/acc_{model}.png')
        plt.clf()
        print(f"Model {model} finished")

    # plot all acc in onr fig
    for i,model in enumerate(models):
        plt.plot(range(400), load_acc(model), label=model, color=color[i])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of All Models")
    plt.legend()

    best_epoch = []
    for i, model in enumerate(models):
        max_x=np.argmax(load_acc(model))
        max_acc=np.max(load_acc(model))

        plt.vlines(max_x,0 , max_acc, colors=color[i], linestyles='dashed')

    plt.savefig(f'img/acc_all.png')
    plt.clf()

    
    for i, model in enumerate(models):
        plot_per_class_accuracy(
            {model: load_best_models(model)},
            test_loader,
            get_label_names(),
            data.get_img_num_per_cls(len(get_label_names()), len(train_dataset), 'exp', 0.01),
            model,
            100,
            'cuda'
        )
    
    


        
