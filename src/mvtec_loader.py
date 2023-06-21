import os
import numpy as np
import glob
import copy

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


class MVTecAD(torch.utils.data.Dataset):

    def __init__(self, root='/home/dataset/mvtec', category='bottle', train: bool=True, transform=None, resize=256, cropsize=224):
        """
        :param root:        MVTecAD dataset dir
        :param category:    MVTecAD category
        :param train:       If it is true, the training mode
        :param transform:   pre-processing
        """
        self.root = root
        self.category = category
        self.train = train
        self.output_size = cropsize

        self.train_dir = os.path.join(root, category, 'train')
        self.test_dir = os.path.join(root, category, 'test')
        self.gt_dir = os.path.join(root, category, 'ground_truth')

        self.normal_class = ['good']
        self.abnormal_class = os.listdir(self.test_dir)
        self.abnormal_class.remove(self.normal_class[0])

        # Setup transform
        if transform is None:
            self.transform_img = transform
            self.transform_img = T.Compose([T.Resize(resize, T.InterpolationMode.BICUBIC),
                                        T.CenterCrop(cropsize),
                                        T.ToTensor(),
                                        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        else:
            self.transform_img = transform
        self.transform_mask = T.Compose([T.ToTensor()])
        self.transform_mask = T.Compose([T.Resize(resize, T.InterpolationMode.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.Resize((224,224)),
                                         T.ToTensor()])

        # Setup dataset path
        if self.train:
            img_paths = glob.glob(os.path.join(
                self.train_dir, self.normal_class[0], '*.png'
            ))
            self.img_paths = sorted(img_paths)
            self.labels = len(self.img_paths) * [0]
            self.gt_paths = len(self.img_paths) * [None]
        else:
            img_paths = []
            labels = []
            gt_paths = []
            for c in os.listdir(self.test_dir):
                paths = glob.glob(os.path.join(
                    self.test_dir, c, '*.png'
                ))
                img_paths.extend(sorted(paths))
                
                if c == self.normal_class[0]:
                    labels.extend(len(paths) * [0])
                    gt_paths.extend(len(paths) * [None])
                else:
                    for i,abclass in enumerate(self.abnormal_class):
                        if c == abclass:
                            labels.extend(len(paths) * [i+1])
                    gt_paths.extend(sorted(glob.glob(os.path.join(self.gt_dir, c, '*.png'))))

            self.img_paths = img_paths
            self.labels = labels
            self.gt_paths = gt_paths
        
        assert len(self.img_paths) == len(self.labels), 'number of x and y should be same'


    def __getitem__(self, index):
        """
        :return:
            original:    original image
            input:  input data to the model
            label:  original image + noise
            mask:   blind spot index
        """
        img_path, target, mask_path = self.img_paths[index], self.labels[index], self.gt_paths[index]

        img = Image.open(img_path).convert('RGB') 
        
        
        if target == 0:
            mask = torch.zeros([1, self.output_size, self.output_size])
        elif target != 0:
            mask = Image.open(mask_path)
            mask = self.transform_mask(mask)

        if self.transform_img:
            img = self.transform_img(img)
            # plt.imsave('test.png',img)
        
        return img, str(target), mask, img_path

    def __len__(self):
        return len(self.img_paths)


def get_mvtec_dataset(category, train_transform=None, test_transform=None, resize=224, cropsize=224):
    train_dataset = MVTecAD(category=category, train=False, transform=train_transform, resize=resize, cropsize=cropsize)
    test_dataset = MVTecAD(category=category, train=False, transform=test_transform, resize=resize, cropsize=cropsize)
    return train_dataset, test_dataset


def get_mvtec_loader(category, train_transform=None, test_transform=None, batch_size=16):
    train_dataset, test_dataset = get_mvtec_dataset(category, train_transform, test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    # self.transform_img = T.Compose([T.Resize(128, Image.ANTIALIAS),
    #                                     T.CenterCrop(112),
    #                                     T.ToTensor(),
    #                                     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                 std=[0.229, 0.224, 0.225])])
    print('setup loader')
    train_loader, test_loader = get_mvtec_loader('bottle')
    for batch in train_loader:
        print(batch[0].size(), batch[1].size(), batch[2].size())
        break
    
    import pdb;pdb.set_trace()
    for batch in test_loader:
        print(batch[0].size(), batch[1].size(), batch[2].size())
        break