# -*- coding: utf-8 -*-
'''
    Author:Zengzhichao
'''

import mxnet as mx
import cv2
import numpy as np
from PIL import Image
import os
from mxnet.gluon.data.vision import transforms
from mxnet import nd

class CUBDataSet_Train(mx.gluon.data.Dataset):
    def __init__(self, path, **kwargs):
        super(CUBDataSet_Train, self).__init__(**kwargs)
        self.root = path
        self.image_path = {}
        self.class_id = {}
        self.train_id = []
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.image_path[img_id] = img_path
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                img_id, cls_id = line.strip().split()
                self.class_id[img_id] = cls_id
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                if int(is_train):
                    self.train_id.append(img_id)
    
    def __getitem__(self, index):   
        img_id = self.train_id[index]
        image = mx.image.imread(os.path.join(self.root, 'images', self.image_path[img_id]))
        #print(image.shape)
        label_id = int(self.class_id[img_id]) - 1
        #image = image[:, :, ::-1]
        #image = cv2.resize(image, (224, 224))
        #image = mx.nd.array(image)
        return image, label_id
        
    def __len__(self):
        return len(self.train_id)
        
class CUBDataSet_Test(mx.gluon.data.Dataset):
    def __init__(self, path, **kwargs):
        self.root = path
        super(CUBDataSet_Test, self).__init__(**kwargs)
        self.image_path = {}
        self.class_id = {}
        self.test_id = []
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.image_path[img_id] = img_path
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                img_id, cls_id = line.strip().split()
                self.class_id[img_id] = cls_id
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                if not int(is_train):
                    self.test_id.append(img_id)
    
    def __getitem__(self, index):
        img_id = self.test_id[index]
        image = mx.image.imread(os.path.join(self.root, 'images', self.image_path[img_id]))
        #print(image.shape)
        label_id = int(self.class_id[img_id]) - 1
        #image = image[:, :, ::-1]
        #image = mx.nd.array(image)
        return image, label_id
        
    def __len__(self):
        return len(self.test_id)
        
def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    
       Args:
            dataset: instance of CUBDataSet_Test,CUBDataSet_Train
    """
    mean_b, mean_g, mean_r, M = 0, 0, 0, 0
    std_b, std_g, std_r, N = 0, 0, 0, 0
    
    for root, dir, file in os.walk(dataset):
        for fi in file:
            #print(fi)
            if os.path.isfile(os.path.join(root,fi)):
                img = cv2.imread(os.path.join(root, fi))
                img = mx.nd.array(img[:,:,::-1])
                mean_b += nd.mean(img[2, :, :])
                mean_g += nd.mean(img[1, :, :])
                mean_r += nd.mean(img[0, :, :])
                M += 1
                #print(img.shape)
    mean_b, mean_g, mean_r = mean_b/M, mean_g/M, mean_r/M
    
    for root, dir, file in os.walk(dataset):
        for fi in file:
            if os.path.isfile(os.path.join(root,fi)):
                img = cv2.imread(os.path.join(root, fi))
                img = mx.nd.array(img[:,:,::-1])
                std_b += nd.sum((img[2:, :, :] - mean_b)**2)
                std_g += nd.sum((img[1:, :, :] - mean_g)**2)
                std_r += nd.sum((img[0:, :, :] - mean_r)**2)
                N = N + nd.prod(nd.array(img[0, :, :].shape), axis=0)
    std_b = nd.sqrt(std_b / N )
    std_g = nd.sqrt(std_g / N )
    std_r = nd.sqrt(std_r / N )
    
    #normalize
    mean_b, mean_g, mean_r = mean_b.asscalar()/255.0, mean_g.asscalar()/255.0, mean_r.asscalar()/255.0
    std_b, std_g, std_r = std_b.asscalar()/255.0, std_g.asscalar()/255.0, std_r.asscalar()/255.0
    
    print('Finshed')
    return (mean_r, mean_g, mean_b),(std_r, std_g, std_b)
    
def test():
    root_dir = os.path.join('./', 'CUB_200_2011')
    #train_trans = transforms.Compose([transforms.Resize((224, 224))])
    train_loader = mx.gluon.data.DataLoader(CUBDataSet_Train(path = root_dir), \
                                            batch_size=256, shuffle=True, num_workers=4, last_batch='keep')
    for idx, batch in enumerate(train_loader):
        print(len(batch))
        print(batch[0])
        break
    '''
        mean :  []
        std  :  [] 
    '''
    mean, std = compute_mean_and_std(train_loader)  
    print(mean, std)
if __name__ == '__main__':
    mean, std = compute_mean_and_std(os.path.join('./', 'CUB_200_2011', 'images'))
    print(mean, std)
