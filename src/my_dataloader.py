from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
from glob import glob
from pathlib import Path
import os
import random
import json

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class shoden_dataset(Dataset): 
    def __init__(self,mode,transform,num_samples=100000): 
        self.transform=transform
        self.mode = mode
       
        self.train_imgs={
            "data":[],
            "caption":[],
        }
       
        self.test_imgs={
            "data":[],
            "caption":[],
        }
        
        with open('/home/oshita/vlm/week_inspector/image/detail.txt') as file:
            lines = file.readlines()
            for line in lines:
                path,caption = line.split('$$')[0],line.split('$$')[1].replace("\n","")
                self.train_imgs["data"].append(path)
                self.train_imgs["caption"].append(caption[0])
        

        p = np.random.permutation(len( self.train_imgs["caption"]))
        self.train_imgs["caption"] = np.array(self.train_imgs["caption"])[p]
        self.train_imgs["data"] = np.array(self.train_imgs["data"])[p]
        
        self.test_imgs["data"] = self.train_imgs["data"][800:810]
        self.test_imgs["caption"] = self.train_imgs["caption"][800:810]
        self.train_imgs["data"] = self.train_imgs["data"][0:800]
        self.train_imgs["caption"] = self.train_imgs["caption"][0:800]
        # self.test_imgs["data"] = self.train_imgs["data"][800:810]
        # self.test_imgs["caption"] = self.train_imgs["caption"][800:810]
        # import pdb;pdb.set_trace()



    def __getitem__(self, index):
        if self.mode=='train':
            
            img = Image.open(self.train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.train_imgs["caption"][index]
        else:
            img = Image.open(self.test_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.test_imgs["caption"][index]
            
        
    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs["data"])
        else:
            return len(self.test_imgs["data"])



class shoden_loader():
    def __init__(self,preprocess):

        self.transform_test = transforms.Compose([
                transforms.ToTensor(),        
                 ])
        if preprocess != None:
            self.transform_train = preprocess
        else:
            self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=224, interpolation=bicubic, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.RandomErasing(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            # Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ])
       
            
        self.train_dataset=shoden_dataset('train',self.transform_train,num_samples=100000)
        self.test_dataset=shoden_dataset('test',self.transform_test,num_samples=100000)
        
        
    def run(self):
        train_loader = DataLoader(
            self.train_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=64,
            shuffle=True,
            num_workers=os.cpu_count()
        )
 
        test_loader = DataLoader(
            self.test_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=2,
            shuffle=True,
            num_workers=os.cpu_count()
        )

        return train_loader,test_loader
        
        

       
if __name__ == '__main__':
    transform=transforms.Compose([
                transforms.ToTensor(),       
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)), ]) 
    loader = shoden_loader(transform)
    train_data,test_data = loader.run()
    import pdb;pdb.set_trace()
   
    