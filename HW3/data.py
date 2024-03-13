import os
import requests, tarfile
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import PIL.Image
import pickle
import matplotlib.pyplot as plt
from utils import get_label_names

CURRENT_DIR = os.getcwd()

CIFAR_DIR = os.path.join(CURRENT_DIR, 'data','cifar')

def download_CIFAR(path, url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', redownload=False):
    if os.path.exists(path) and redownload is False:
        print('Dataset already exists')
        return
    print('Downloading...')
    r = requests.get(url, stream=True)
    # show download progress
    total_length = r.headers.get('content-length')
    with open('cifar-100-python.tar.gz', 'wb') as f:
        if total_length is None: # no content length header
            f.write(r.content)
        else:
            dl = 0
            total_length = int(total_length)
            for chunk in r.iter_content(chunk_size=1024):
                dl += len(chunk)
                f.write(chunk)
                done = int(50 * dl / total_length)
                print('\r[{}{}]'.format('=' * done, ' ' * (50-done)), end='')
    print('Extracting...')
    with tarfile.open('cifar-100-python.tar.gz') as f:
        f.extractall(path)
    print('Done!')

def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    img_max = total_num / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_data(img_num_per_cls, imgList, labelList):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    new_data = []
    new_targets = []
    targets_np = np.array(labelList, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)  # remove shuffle in the demo fair comparision
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        #np.random.shuffle(idx) # remove shuffle in the demo fair comparision
        selec_idx = idx[:the_img_num]
        new_data.append(imgList[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return (new_data, new_targets)



class CIFAR100LT(Dataset):
    def __init__(self, set_name='train', imageList=[], labelList=[], labelNames=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        self.labelNames = labelNames
        if self.set_name=='train':            
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        self.imageList = imageList
        self.labelList = labelList
        self.current_set_len = len(self.labelList)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx]
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage = self.transform(curImage)    

        curLabel =  np.asarray(self.labelList[idx])
        curLabel = torch.from_numpy(curLabel.astype(np.int64))
        return curImage, curLabel
    

def get_CIFAR100LT_data(type:str):
    labelnames = get_label_names()

    with open(os.path.join(CIFAR_DIR, 'cifar-100-python', type), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'fine_labels']

    total_num = len(labelList)
    if type == 'train':
        img_num_per_cls = get_img_num_per_cls(len(labelnames), total_num, 'exp', 0.01)
        new_imgList, new_labelList = gen_imbalanced_data(img_num_per_cls, imgList, labelList)

        plt.plot(img_num_per_cls)
        plt.xlabel('class ID sorted by cardinality')
        plt.ylabel('#training examples')

        plt.savefig('imbalanced_cifar100_train.png')

        return CIFAR100LT(
            imageList=new_imgList, labelList=new_labelList, labelNames=labelnames,
            set_name=type, isAugment=type=='train')
    
    return CIFAR100LT(
        imageList=imgList, labelList=labelList, labelNames=labelnames,
        set_name=type, isAugment=type=='train')


if __name__ == '__main__':
    data_sampler=iter(get_CIFAR100LT_data('train'))
    data = next(data_sampler)
    imageList, labelList = data

    device='cuda'
    imageList = imageList.to(device)
    labelList = labelList.type(torch.long).view(-1).to(device)

    print(imageList.shape)
    print(labelList.shape)

