import os
#from PIL import Image
import shutil
import json

count = 0
anno_list = {}
for _, __, fs in os.walk('/home/lance/Desktop/101/food class/fooddata/train'):
	for d in __:
		anno_list[d] = count
		count = count+1
		if d.split(" ") != d:
			for aa in d.split(" "):
				if aa not in anno_list:
					anno_list[aa] = count
					count = count + 1
json.dump(anno_list, open("data/fooddata/data/food_cato.json", 'w'))

acc=[]
for _, __, fs in os.walk('/home/lance/Desktop/graph/ML_GCN-master/data/fooddata/data/train'):
	for d in __:
		for pp,dd, ff in os.walk(_+'/'+d):
				for fs in ff:
					temp={}
					temp["file_name"] = pp + '/' + fs
					labels=[]

					for ttt in anno_list:
						if ttt in d:
							labels.append(anno_list[ttt])
					temp["labels"] = labels
					acc.append(temp)
json.dump(acc, open("data/fooddata/data/foodtrain.json", 'w'))


acc=[]
for _, __, fs in os.walk('/home/lance/Desktop/graph/ML_GCN-master/data/fooddata/data/test'):
	for d in __:
		for pp,dd, ff in os.walk(_+'/'+d):
				for fs in ff:
					temp={}
					temp["file_name"] = pp + '/' + fs
					labels=[]

					for ttt in anno_list:
						if ttt in d:
							labels.append(anno_list[ttt])
					temp["labels"] = labels
					acc.append(temp)
json.dump(acc, open("data/fooddata/data/foodval.json", 'w'))

'''
count = 0
anno_list = {}
for _, __, fs in os.walk('/home/lance/Desktop/101/food class/fooddata/data/train'):
	for d in __:
		anno_list[d] = count
		count = count+1
		if d.split(" ") != d:
			for aa in d.split(" "):
				anno_list[aa] = count
				count = count + 1
json.dump(anno_list, open("data/fooddata/food_cato.json", 'w'))
'''

'''
for _, __, fs in os.walk('/home/lance/Desktop/101/food class/fooddata/test'):
    for d in __:
        a = d.replace("_", " ")
        shutil.move(_ + '/' + d, _ + '/' + a)

for _, __, fs in os.walk('/home/lance/Desktop/101/food class/fooddata/val'):
    for d in __:
        a = d.replace("_", " ")
        shutil.move(_ + '/' + d , _ + '/' + a)
'''
