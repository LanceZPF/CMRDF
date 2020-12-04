import os
#from PIL import Image
import shutil
import json
import pickle

tempa = open("data/voc/voc_glove_word2vec.pkl","rb")

a = pickle.load(tempa)
print(a)

'''
a = open("glove.6B.300d.txt")
a.read()

print(len(a))

print(len(a[0]))

f = open("temp.txt","w")
f.write(str(a.tolist()))
f.close()
'''
'''
aaa={}

acc=[]
for ii in a.values():
	#print(ii)
	account = 0
	for iii in b:
		if ii in iii["labels"]:
			account=account+1
	acc.append(account)

aaa['nums']=np.array(acc)

abb=[]
for i in a.values():
	aqq=[]
	for ii in a.values():
		account=0
		for iii in b:
			if i in iii["labels"] and ii in iii["labels"]:
				account=account+1
		aqq.append(account)
	abb.append(aqq)

aaa['adj']=np.array(abb)

print(abb)

with open("data/fooddata/food_adj.pkl","wb") as f:
	pickle.dump(aaa,f)


'''
