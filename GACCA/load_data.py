from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        #print(len(self.images))
        #print(len(self.labels))
        assert len(
            self.images) == len(self.labels)
        return count


def ind2vec(ind, N=None):
    ppp =[]
    #print(ind[0])
    #print(max(ind[0]))
    ind = np.asarray(ind[0])
    #a = int(max(ind))
    a = 20181
    for i in range(len(ind)):
        ttt = []
        ttt.append(ind[i])
        ppp.append(ttt)
    ppp = np.asarray(ppp)
    #print(ppp)
    #print(ind)
    #if N is None:
    #   N = ind.max() + 1
    #print(N.size())
    #print(np.arange(N))
    #print(np.repeat(ind, N, axis=1))
    #return np.arange(N) == np.repeat(ind, N, axis=1)
    #print(len(np.arange(N)))
    #print(np.repeat(ppp,a,axis=1))
    return np.repeat(ppp,a,axis=1)

def get_loader(path, batch_size):
    img_train = loadmat(path+"train_img.mat")['train_img']
    #print(len(img_train))
    img_test = loadmat(path + "test_img.mat")['test_img']
    text_train = loadmat(path+"train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
    label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']


    label_train = ind2vec(label_train)
    label_test = ind2vec(label_test)
    #print(len(img_train))
    #print(len(img_test))
    #print(len(label_train[0]))
    #print(len(label_test[0]))

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test']}

    shuffle = {'train': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    #print(text_train)
    #print(label_train)
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par

