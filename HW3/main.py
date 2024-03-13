import data
import torch
from model import CNN, training
from pytorch_cifar.models.resnet import ResNet, BasicBlock, Bottleneck
from pytorch_cifar.models.mobilenetv2 import MobileNetV2

train_dataset= data.get_CIFAR100LT_data('train')
test_dataset = data.get_CIFAR100LT_data('test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

model=CNN()
model.to('cuda')

model=training(
    model=model,
    trainloader=train_loader,
    testloader=test_loader,
    name='CNN',
)

model=ResNet(BasicBlock, [2, 2, 2, 2],100)
model.to('cuda')

model=training(
    model=model,
    trainloader=train_loader,
    testloader=test_loader,
    name='ResNet18',
)

model=ResNet(Bottleneck, [3, 4, 6, 3],100)
model.to('cuda')

model=training(
    model=model,
    trainloader=train_loader,
    testloader=test_loader,
    name='ResNet50',
)

model=ResNet(Bottleneck, [3, 8, 36, 3],100)
model.to('cuda')

model=training(
    model=model,
    trainloader=train_loader,
    testloader=test_loader,
    name='ResNet152',
)

model=MobileNetV2(num_classes=100)
model.to('cuda')

model=training(
    model=model,
    trainloader=train_loader,
    testloader=test_loader,
    name='MobileNetV2',
)





