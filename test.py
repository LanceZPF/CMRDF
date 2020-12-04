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

path = ["./puzzle_devset/001/images","./puzzle_devset/002/images","./puzzle_devset/003/images","./puzzle_devset/004/images","./puzzle_devset/005/images","./puzzle_devset/006/images","./puzzle_devset/007/images","./puzzle_devset/008/images","./puzzle_devset/009/images","./puzzle_devset/010/images"]
save_path="./selectedIM"
tempa = open("./category.json","r")
a = json.load(tempa)

for i in range(10):
    files = os.listdir(path[i])
    for filename in files:
        #print(filename)
        filename2=path[i]+'/'+filename
        aaa = Image.open(filename2).convert('RGB')
        aaa = transform(aaa)
        aaa = aaa.unsqueeze(0)
        aaa = aaa.to(device)
        with torch.no_grad():
            py = model(aaa, inp)
            # print(model(auto, inp).data)
        
        cc=0
        for ii in py[0]:
            if float(ii) > 0:
                for iii in a.values():
                    if cc == iii:
                        #print(filename,end=" ")
                        print(list(a.keys()) [list (a.values()).index(iii)],end=" ")
                        print(float(ii))   
                img=Image.open(filename2)
                img.save(save_path+"/"+str(i)+"/"+filename)           
            cc=cc+1
        
        print("\n")

