import os
#from PIL import Image
import torch
import shutil
import json
import pickle
import numpy as np


def load(filename):
	# Input: GloVe Model File
	# More models can be downloaded from http://nlp.stanford.edu/projects/glove/
	# glove_file="glove.840B.300d.txt"
	glove_file = filename

	dimensions = 300

	num_lines = getFileLineNums(filename)
	# num_lines = check_num_lines_in_glove(glove_file)
	# dims = int(dimensions[:-1])
	dims = 300

	print(num_lines)
	#
	# # Output: Gensim Model text format.
	gensim_file = 'glove_model.txt'
	gensim_first_line = "{} {}".format(num_lines, dims)
	#
	# # Prepends the line.
	prepend_line(glove_file, gensim_file, gensim_first_line)

	# Demo: Loads the newly created glove_model.txt into gensim API.
	model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model

	model_name = filename[5:-4]

	model.save(model_name)

	return model


def getFileLineNums(filename):
	f = open(filename, 'r')
	count = 0

	for line in f:
		count += 1
	return count


def prepend_line(infile, outfile, line):
	"""
    Function use to prepend lines using bash utilities in Linux.
    (source: http://stackoverflow.com/a/10850588/610569)
    """
	with open(infile, 'r') as old:
		with open(outfile, 'w') as new:
			new.write(str(line) + "\n")
			shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
	"""
    Slower way to prepend the line by re-creating the inputfile.
    """
	with open(infile, 'r') as fin:
		with open(outfile, 'w') as fout:
			fout.write(line + "\n")
			for line in fin:
				fout.write(line)

tempa = open("data/fooddata/data/category.json","r")
tempb = open("data/fooddata/data/val_anno.json")

a = json.load(tempa)
b = json.load(tempb)


aaa={}

acc=[]
for ii in a.values():
	#print(ii)
	account = 0
	for iii in b:
		if ii in iii["labels"]:
			account=account+1
	acc.append(account)

aaa['nums']=torch.Tensor(acc)

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

aaa['adj']=torch.Tensor(abb)

#print(abb)

with open("data/fooddata/food_adj.pkl","wb") as f:
	pickle.dump(aaa,f)


import gensim
#load("glove.6B.300d.txt")
model=gensim.models.KeyedVectors.load_word2vec_format('glove_model.txt',binary=False)

alist=[]
#try:
for sss in a:
	temp = [0] * 300
	if len(sss.split(" ")) == 1:
		if sss =="macarons":
			temp = model["dessert"]
		else:
			temp = model[sss]
	else:
		numss = 0
		for aaa in sss.split(" "):
			numss = numss + 1
			if aaa == "macarons":
				temp = temp + model["dessert"]
			else:
				temp = temp + model[aaa]
		temp = temp/numss
	alist.append(temp)
alist=torch.Tensor(alist)
#except:
#	print("aaaaa: "+ sss )
with open("data/fooddata/food_glove_word2vec.pkl","wb") as ff:
	pickle.dump(alist,ff)