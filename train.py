import torch
import torch.nn as nn
import clip

import math
import random
import numpy as np
from tqdm import tqdm
import argparse
from src.my_dataloader import shoden_loader as loader
from src.cifar_loader import cifar_loader
import src.MlflowWriter as MW
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER
# from src.eva_clip import create_eva_vit_g
# from lavis.models import load_model
"""準備"""

parser = argparse.ArgumentParser(description='ハイパラに関して')
parser.add_argument('--dataset', help = 'use data',type=str,default='my_dataset')
parser.add_argument('--model', help = 'use data',type=str,default='ViT-B/32',choices=clip.available_models())
parser.add_argument('--epoch', type=int,default=10)
parser.add_argument('--optimizer',type=str,default='Adam')
parser.add_argument('--scheduler',type=str,default='exponential')
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--seed',type=int,default=9999)
args = parser.parse_args()


EXPERIMENT_NAME = 'CLIP_Finetuning'
writer = MW.MlflowWriter(EXPERIMENT_NAME)
tags = {'trial':args.seed,
        'epoch':args.epoch,
        MLFLOW_RUN_NAME:f'tag:{args.dataset}_{args.optimizer}_{args.seed}_{args.lr}_{args.scheduler}',
        MLFLOW_USER:args.model}
writer.create_new_run(tags)
writer.log_param('epoch',args.epoch)

"""### シード値の固定"""
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)


"""学習データ，モデル定義"""
device = torch.device('cuda')
model,preprocess = clip.load('ViT-B/32',device=device,jit=False) 
clip.model.convert_weights(model)

model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text)
model_image = torch.nn.DataParallel(model_image)

optimizer = torch.optim.Adam(model.parameters(), args.lr)
# scheduller = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.8,last_epoch=-1)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# train_data,test_data = loader(preprocess).run()
loader = cifar_loader('/home/dataset/cifar10',256)
train_data,valid,test = loader.get_loader()



"""学習"""
# torch.autograd.set_detect_anomaly(True)

for epoch in range(10):
    model.train()
    loop = tqdm(train_data, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))
    scaler = torch.cuda.amp.GradScaler()
    batch_loss = []

    for batch in loop:
        optimizer.zero_grad()
        images,text = batch
        text = clip.tokenize(text)
        # with torch.cuda
        image_embedding = model_image(images)
        text_embedding = model_text(text)


        logit_scale = model.logit_scale.exp()
        logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
        ground_truth = torch.arange(len(images)).to(device) # this one still need manually to put on GPU  

        # total_loss = torch.Tensor(0,requires_grad = True)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.retain_grad()
        total_loss.backward()
        print(total_loss)
        print(total_loss.grad)
        
        
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

        batch_loss.append(total_loss)

    train_avg_loss = torch.tensor(batch_loss).mean()
    writer.log_metric_step('contrastive_loss',train_avg_loss,epoch)
    print(f"| Train | Epoch   {epoch+1} |: contrastive_loss:{train_avg_loss:.3f}")
print('Finished Training')
writer.set_terminated()
        
