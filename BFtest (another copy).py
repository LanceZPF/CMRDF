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
countnum=0

testconf=[]

testhash=[]

testfeature=[]

testsemantic=[]

testweight=[]

testmatrix=[]

haha = model.features
haha.eval()
pooling = torch.nn.MaxPool2d(14,14)

tempa = open("./category.json","r")
a = json.load(tempa)

out=open("labels.csv","w",newline="")
cw=csv.writer(out,dialect='excel')

pbar=tqdm(total=3000)

for pathe, dirs, fs in os.walk("RLD"):
	for f in fs:
		aaa = Image.open("RLD/"+f).convert('RGB')
		aaa = transform(aaa)

		aaa = aaa.unsqueeze(0)
		aaa = aaa.to(device)

		with torch.no_grad():
			py = model(aaa, inp)
			feature_maps=haha(aaa)
			feature_vector=pooling(feature_maps)
			feature_vector.squeeze()
			if True:
				testconf.append(py[0])
				testfeature.append(feature_vector)

		pbar.update(1)
		preconf=py[0]
		cc=0
		outlist=[]
		outlist.append(f)

		hashi=torch.tensor([]).to(device)
		seman=torch.tensor([]).to(device)
		weigh=torch.tensor([]).to(device)
		truem=torch.tensor([]).to(device)

		for ii in range(len(preconf)):
			if float(preconf[ii]) > 0.4:

				truem=torch.cat((truem,inp[0][ii]),0)
				cc=cc+1
		
		if cc == 0:
			truem=torch.cat((truem,inp[0][pretemp]),0)
		
		if True:
			testhash.append(hashi)
			testsemantic.append(seman)
			testweight.append(weigh)
			testmatrix.append(truem)

		cw.writerow(f)
		countnum=countnum+1


with open("./test_conf.pkl","wb") as f:
	pickle.dump(testconf,f)

with open("./test_hash.pkl","wb") as f:
	pickle.dump(testhash,f)

with open("./test_feature.pkl","wb") as f:
	pickle.dump(testfeature,f)

with open("./test_semantic.pkl","wb") as f:
	pickle.dump(testsemantic,f)

with open("./test_weight.pkl","wb") as f:
	pickle.dump(testweight,f)

with open("./test_spp.pkl","wb") as f:
	pickle.dump(testmatrix,f)

pbar.close()
out.close()
