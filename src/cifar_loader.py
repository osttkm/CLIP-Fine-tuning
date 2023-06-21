from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os


class cifar_dataset(Dataset): 

    """シード値の固定"""
    random_seed = 9999
    np.random.seed(random_seed)
    def __init__(self, root, mode,transform,num_samples=0): 
        self.root=root
        self.mode=mode
        self.transform=transform

        imgs={
            "data":[],
            "label":[],}

        self.train_imgs={
            "data":[],
            "label":[],}
        self.valid_imgs={
            "data":[],
            "label":[],}
        self.test_imgs={
            "data":[],
            "label":[],}

        # ここは自分で設定すること．cifarのtrain,testの画像ファイルがあるディレクトリを指定
        data_root = Path(self.root)

        # 別途trainとtestのパスを作る
        train_path = data_root / "train"
        test_path = data_root / "test"

        # train,testディレクトリのそれぞれのクラスのディレクトリを作成
        train_dir = [path for path in train_path.iterdir() if path.is_dir()]
        test_dir = [path for path in test_path.iterdir() if path.is_dir()]

        for category_dir in train_dir:
            label = category_dir.name
            img_paths = list(category_dir.glob("*.png"))
            img_num = len(img_paths)
            for path in img_paths:
                imgs['data'].append(str(path))
                if int(path.parent.name)==0: name = 'plane'
                elif int(path.parent.name)==1: name = 'car'   
                elif int(path.parent.name)==2: name = 'bird'
                elif int(path.parent.name)==3: name = 'cat'
                elif int(path.parent.name)==4: name = 'deer'
                elif int(path.parent.name)==5: name = 'dog'
                elif int(path.parent.name)==6: name = 'flog'
                elif int(path.parent.name)==7: name = 'horse'
                elif int(path.parent.name)==8: name = 'ship'    
                elif int(path.parent.name)==9: name = 'bus' 
                # imgs['label'].append(int(path.parent.name))
                imgs['label'].append(name)

        for category_dir in test_dir:
            label = category_dir.name
            img_paths = list(category_dir.glob("*.png"))
            img_num = len(img_paths)
            for path in img_paths:
                self.test_imgs['data'].append(str(path))
                self.test_imgs['label'].append(int(path.parent.name))

        # 同じ並び替え方でシャッフル
        p = np.random.permutation(len(imgs['label']))
        imgs['label'] = np.array(imgs['label'])[p]
        imgs['data'] = np.array(imgs['data'])[p]
       
        # 各クラス数が等しくなるように配列に追加
        for _cls in np.unique(imgs['label']):
            self.train_imgs['data'].extend(imgs['data'][imgs['label']==_cls][0:4000])
            self.train_imgs['label'].extend(imgs['label'][imgs['label']==_cls][0:4000])
            self.valid_imgs['data'].extend(imgs['data'][imgs['label']==_cls][4000:])
            self.valid_imgs['label'].extend(imgs['label'][imgs['label']==_cls][4000:])
        
                    
    def __getitem__(self, index):
        if self.mode=='train':
            img = Image.open(self.train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.train_imgs["label"][index]
        if self.mode=='valid':
            img = Image.open(self.valid_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.valid_imgs["label"][index]
        else:
            img = Image.open(self.test_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.test_imgs["label"][index]
        
    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs["data"])
        if self.mode=="valid":
            return len(self.valid_imgs["data"])
        else:
            return len(self.test_imgs["data"])



class cifar_loader():
    def __init__(self,root,batch_size):
        self.batch_size = batch_size
        self.root = root

        # ここは自分で色々変える
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.CenterCrop(size=(224, 224)),
            # transforms.RandomErasing(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)), ])     

        self.test_transform= transforms.Compose([
                transforms.ToTensor(),                
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),  ])
  
        self.train_dataset=cifar_dataset(self.root,'train',self.train_transform,num_samples=50000)
        self.valid_dataset=cifar_dataset(self.root,'valid',self.test_transform,num_samples=50000)
        self.test_dataset=cifar_dataset(self.root,'test',self.test_transform,num_samples=50000)
  
    def get_loader(self):
        train_loader = DataLoader(
            self.train_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count())
        valid_loader = DataLoader(
            self.valid_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count())
        test_loader = DataLoader(
            self.test_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count())
        return train_loader,valid_loader,test_loader

if __name__ == '__main__':
    loader=cifar_loader('/home/dataset/cifar10',256)
    train,valid,test = loader.get_loader()