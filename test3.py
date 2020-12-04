# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:01:24 2019

@author: xia
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from models import *
from util import *
import pickle
import json
from tqdm import tqdm
import os

#print(inp['nums'].size())
#print(inp['adj'].size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gcn_resnet101(num_classes=199, t=0.4, adj_file='food_adj.pkl')
model.load_state_dict(torch.load("model_best_86.4951.pth.tar")['state_dict'])
model = model.to(device)
model.eval()
normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                std=model.image_normalization_std)
transform = transforms.Compose([
                Warp('448'),
                transforms.ToTensor(),
                normalize,
            ])
fff = open('food_glove_word2vec.pkl','rb')
inp = pickle.load(fff)
inp = torch.Tensor(inp)
inp = inp.unsqueeze(0)
inp = inp.to(device)

save_path="./selectedIM6"

path="./u2"
rule=".JPG"

for fpathe,dirs,fs in os.walk(path):   # os.walk是获取所有的目录
    for f in fs:
        filename = os.path.join(fpathe,f)
        try:
            if filename.endswith(rule):  # 判断是否是"xxx"结尾
                aaa = Image.open(filename).convert('RGB')
                aaa = transform(aaa)
                aaa = aaa.unsqueeze(0)
                aaa = aaa.to(device)
                with torch.no_grad():
                    py = model(aaa, inp)
            # print(model(auto, inp).data)
                flag=0
                for ii in py[0]:
                    if float(ii) > -2:
                        flag=flag+1
                   # img=Image.open(filename)
                  #  img.save(save_path+"/"+f)     
                if flag>0:
                    img=Image.open(filename)
                    img.save(save_path+"/"+f)
        except:
            print("+1") 
                #print("+1")    
                

