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
from src.imagenet1k_loader import imagenet1k_loader
import src.MlflowWriter as MW
from src.util import *
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER
from src.eva_vit import create_eva_vit_g
# from lavis.models import load_model
"""準備"""

parser = argparse.ArgumentParser(description='ハイパラに関して')
parser.add_argument('--dataset', help = 'use data',type=str,default='my_dataset')
parser.add_argument('--model', help = 'use data',type=str,default='ViT-B/32',choices=clip.available_models())
parser.add_argument('--epoch', type=int,default=10)
parser.add_argument('--optimizer',type=str,default='Adam')
parser.add_argument('--scheduler',type=str,default='cosine')
parser.add_argument('--warmup_lr_times',type=float,default = 0.1)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--seed',type=int,default=9999)
parser.add_argument('--batch_size',type=int,default=256)
args = parser.parse_args()


EXPERIMENT_NAME = 'CLIP_Finetuning'
writer = MW.MlflowWriter(EXPERIMENT_NAME)
tags = {'trial':args.seed,
        'epoch':args.epoch,
        MLFLOW_RUN_NAME:f'tag:{args.batch_size}_{args.optimizer}_{args.seed}_{args.lr}_{args.scheduler}',
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
print('============installing CLIP============')
model,preprocess = clip.load('ViT-B/32',device=device,jit=False) 

# import pdb;pdb.set_trace()
print('============Finish============')

clip.model.convert_weights(model)

model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text)
model_image = torch.nn.DataParallel(model_image)

optim = torch.optim.Adam(model.parameters(), args.lr)
sche = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epoch,eta_min=args.lr*0.01)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# train_data,test_data = loader(preprocess).run()
loader = imagenet1k_loader('/home/dataset/imagenet_2012/',args.batch_size)
train_data,val = loader.get_loader()

early_stopping = EarlyStopping(patience=10,verbose=1)



"""学習"""
# torch.autograd.set_detect_anomaly(True)

for epoch in range(args.epoch):
    if epoch==0: print('============Start Training===========')
    model.train()
    loop = tqdm(train_data, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))
    scaler = torch.cuda.amp.GradScaler()
    batch_loss = []
    best_loss = 0

    if epoch<10:
        _lr = create_lr(epoch,args.lr,args.warmup_lr_times)
        optimizer = torch.optim.Adam(model.parameters(), _lr)
    else:
        optimizer = optim
        scheduller = sche

    for batch in loop:
        optimizer.zero_grad()
        images,text = batch
        text = clip.tokenize(text)

        with torch.cuda.amp.autocast():
            image_embedding = model_image(images)
            text_embedding = model_text(text)


        logit_scale = model.logit_scale.exp()
        logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
        ground_truth = torch.arange(len(images)).to(device) # this one still need manually to put on GPU  

        
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        #　DDPのためにモデルをいじったのでlossがleaf tensor扱いにならないらしい．無理やりleaf tensorに変更
        total_loss.retain_grad()
        # total_loss.backward()
        scaler.scale(total_loss).backward()
                
        convert_models_to_fp32(model)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        clip.model.convert_weights(model)

        batch_loss.append(total_loss)

    train_avg_loss = torch.tensor(batch_loss).mean()
    writer.log_metric_step('contrastive_loss',train_avg_loss,epoch)
    print(f"| Train | Epoch   {epoch+1} |: contrastive_loss:{train_avg_loss:.3f}")

    with torch.no_grad():
        model.eval()
        loop = tqdm(val, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))
        for batch in loop:
            data,text = batch
            batch_loss = []
            text = clip.tokenize(text)

            image_embedding = model_image(images)
            text_embedding = model_text(text)

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
            ground_truth = torch.arange(len(images)).to(device) # this one still need manually to put on GPU 
        
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2  
        batch_loss.append(total_loss) 
    val_avg_loss = torch.tensor(batch_loss).mean()
    writer.log_metric_step('val_contrastive_loss',val_avg_loss,epoch)
    print(f"| Valid | Epoch   {epoch+1} |: contrastive_loss:{val_avg_loss:.3f}")
    if epoch==0:
        print('initial save')
        best_loss = val_avg_loss
        torch.save(model, f'./model_{args.batch_size}_{args.lr}_{args.epoch}.pth')
    elif best_loss > val_avg_loss:
        best_loss = val_avg_loss
        print('update!!')
        torch.save(model, f'./model_{args.batch_size}_{args.lr}_{args.epoch}.pth')
        
    if early_stopping.validate(val_avg_loss):
        print('============Early Stopping============')
        break




print('Finished Training')
writer.set_terminated()
        
