from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import os
import pickle
import random


class imagenet1k_dataset(Dataset): 

    """シード値の固定"""
    random_seed = 9999
    np.random.seed(random_seed)
    def __init__(self, root, mode,transform,num_samples=0): 
        self.root=root
        self.mode=mode
        self.transform=transform

        self.squares = []
        self.square_size = 224 // 3
        self.color = np.array([(255,0,0),(128,128,0),(255,255,0),(255,0,255),(192,192,192),(0,255,255),(0,255,0),(0,128,0),(128,128,128),(0,0,255)])
        self.color_name = np.array(['red','olive','yellow','fuchsia','silver','aqua','lime','green','gray','blue'])
        self.circle_place = np.array(['upper left','left','lower left','upper','center','lower','upper right','right','lower right'])
        
        for i in range(3):
            for j in range(3):
                left = i * self.square_size
                upper = j * self.square_size
                right = left + self.square_size
                lower = upper + self.square_size
                self.squares.append((left, upper, right, lower))
        self.squares = np.array(self.squares)
        

        self.train_imgs={
            "data":[],
            "label":[],}
        self.val_imgs={
            "data":[],
            "label":[],}

        train_path = Path(self.root+'ILSVRC2012_img_train/')
        train_dir = [path for path in train_path.iterdir() if path.is_dir()]

        val_path = Path(self.root+'ILSVRC2012_img_val/')
        val_dir = [path for path in val_path.iterdir() if path.is_dir()]

        class_dict = self.get_class_name()

        for category_dir in train_dir:
            img_paths = [str(path) for path in category_dir.glob('*.JPEG')]
            self.train_imgs['data'].extend(img_paths)
            self.train_imgs['label'].extend([class_dict[category_dir.name]]*len(img_paths))

        for category_dir in val_dir:
            img_paths = [str(path) for path in category_dir.glob('*.JPEG')]
            self.val_imgs['data'].extend(img_paths)
            self.val_imgs['label'].extend([class_dict[category_dir.name]]*len(img_paths))

        p = np.random.permutation(len(self.train_imgs['label']))
        self.train_imgs['label'] = np.array(self.train_imgs['label'])[p]
        self.train_imgs['data'] = np.array(self.train_imgs['data'])[p]

        p = np.random.permutation(len(self.val_imgs['label']))
        self.val_imgs['label'] = np.array(self.val_imgs['label'])[p]
        self.val_imgs['data'] = np.array(self.val_imgs['data'])[p]

                    
    def __getitem__(self, index):
        if self.mode=='train':
            img = Image.open(self.train_imgs["data"][index]).convert('RGB')
            img,caption = self.captioning(img,self.train_imgs["label"][index])
            img = self.transform(img)
            return img,caption
        if self.mode=='val':
            img = Image.open(self.val_imgs["data"][index]).convert('RGB')
            img,caption = self.captioning(img,self.val_imgs["label"][index])
            img = self.transform(img)
            return img,caption
       
    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs["data"])
        if self.mode=="val":
            return len(self.val_imgs["data"])

    def get_class_name(self,txt_path='/home/dataset/imagenet_2012/map_clsloc.txt'):
        data_dict = {}
        with open(txt_path, 'r') as file:
            for line in file:
                # 行をキーと値に分割する（例：キーと値がタブで区切られている場合）
                key,_,value = line.strip().split(' ')
                # 辞書に追加する
                data_dict[key] = value
        return data_dict
        

    def captioning(self,image,label):
        selected_square = random.choice(self.squares)
        circle_color = random.choice(self.color)
        place = self.circle_place[(self.squares==selected_square).sum(axis=1)==4][0]

        # ランダムな半径を生成
        min_radius = self.square_size // 4
        max_radius = self.square_size // 2
        circle_radius = random.randint(min_radius, max_radius)

        # image = Image.open(str(b_image))
        image = image.resize((224,224))
        circle_center = (
            selected_square[0] + self.square_size // 2,
            selected_square[1] + self.square_size // 2
        )
        draw = ImageDraw.Draw(image)
        draw.ellipse((circle_center[0]-circle_radius, circle_center[1]-circle_radius,
                    circle_center[0]+circle_radius, circle_center[1]+circle_radius),
                    outline=tuple(circle_color), width=5)
        
        caption_type1 = f'''A photo of a {self.color_name[(circle_color==self.color).sum(axis=1)==3][0]} circle drawn at the {place} part of the {label} image.'''
        caption_type2 = f'''The {self.color_name[(circle_color==self.color).sum(axis=1)==3][0]} circle is drawn in the {place} part of the {label} image.'''
        
        # print(f'caption1:{caption_type1}')
        # print(f'caption2:{caption_type2}')
        caption = random.choice([caption_type1,caption_type2])
        return image,caption


class imagenet1k_loader():
    def __init__(self,root,batch_size):
        self.batch_size = batch_size
        self.root = root

        # ここは自分で色々変える
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.CenterCrop(size=(224,224)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)), ])     

        self.test_transform= transforms.Compose([
                transforms.ToTensor(),                
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),  ])
  
        self.train_dataset=imagenet1k_dataset(self.root,'train',self.train_transform,num_samples=50000)
        self.val_dataset=imagenet1k_dataset(self.root,'val',self.train_transform,num_samples=50000)
  
    def get_loader(self):
        train_loader = DataLoader(
            self.train_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count())
        val_loader = DataLoader(
            self.val_dataset,
            pin_memory=True,
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count())
        return train_loader,val_loader


def denormalize(tensor):
    mean=(0.48145466, 0.4578275, 0.40821073)
    std=(0.26862954, 0.26130258, 0.27577711)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

if __name__ == '__main__':
    
    loader = imagenet1k_loader('/home/dataset/imagenet_2012/',1024)
    train,val = loader.get_loader()
    train_data0 = train.dataset[0]
    train_data1 = train.dataset[1]
    
    val_data0 = val.dataset[0]
    val_data1 = val.dataset[1]

    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    train_image0 = to_pil(denormalize(train_data0[0]))
    val_image0 = to_pil(denormalize(val_data0[0]))

    train_image1 = to_pil(denormalize(train_data1[0]))
    val_image1 = to_pil(denormalize(val_data1[0]))

    train_image0.save('./train_image0.jpeg')
    val_image0.save('./val_image0.jpeg')
    train_image1.save('./train_image1.jpeg')
    val_image1.save('./val_image1.jpeg')
    import pdb;pdb.set_trace()