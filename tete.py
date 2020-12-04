import torch
from PIL import Image
import torchvision.transforms as transforms
from models import *
from util import *
import pickle
import json
import csv
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

#print(inp['nums'].size())
#print(inp['adj'].size())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gcn_resnet101(num_classes=150, t=0.4, adj_file='food_adj.pkl')
model.load_state_dict(torch.load("model_best_87.7752.pth.tar")['state_dict'])
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
'''

tempa = open("./train_anno.json","r")
tempb = open("./val_anno.json","r")

a = json.load(tempa)
b = json.load(tempb)
qqq=[]
www=[]

with tqdm(total=len(a)) as pbar:
    for aaaa in a:
        aaa = Image.open("./coco/data/train2014/"+aaaa['file_name']).convert('RGB')
        aaa = transform(aaa)
        aaa = aaa.unsqueeze(0)
        aaa = aaa.to(device)
        auto = torch.autograd.Variable(aaa).float().detach()
        pbar.update(1)

        #print(aaa.size())
        #print(auto.size())
        with torch.no_grad():
            py = model(auto, inp)
            #print(model(auto, inp).data)
            qqq.append(py[0])
with open("./train.pkl","wb") as f:
    pickle.dump(qqq,f)

with tqdm(total=len(b)) as pbar:
    for aaaa in b:
        aaa = Image.open("./coco/data/val2014/"+aaaa['file_name']).convert('RGB')
        aaa = transform(aaa)
        aaa = aaa.unsqueeze(0)
        aaa = aaa.to(device)
        auto = torch.autograd.Variable(aaa).float().detach()
        pbar.update(1)

        #print(aaa.size())
        #print(auto.size())
        with torch.no_grad():
            py = model(auto, inp)
            #print(model(auto, inp).data)
            www.append(py[0])
with open("./test.pkl","wb") as f:
    pickle.dump(www,f)
'''



haha = model.features
haha.eval()

pooling = torch.nn.MaxPool2d(14,14)
tempa = open("./category.json","r")
a = json.load(tempa)



aaa = Image.open("1.JPG").convert('RGB')
aaa = transform(aaa)
#print(aaa.numpy())

aaa = aaa.unsqueeze(0)
aaa = aaa.to(device)

with torch.no_grad():
	py = model(aaa, inp)
	feature_maps=haha(aaa)
	feature_vector=pooling(feature_maps)
	feature_vector.squeeze()
	#plt.imshow(feature_maps[0][0].data.cpu().numpy())
	#plt.show()
	#matplotlib.use('TkAgg')
	#print(py)
	#print(feature_maps.shape)
	#plt.imshow(feature_maps[0][0].data.cpu().numpy())
	#plt.show()
	#matplotlib.use('TkAgg')
	#print(py)

