import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
# from sklearn.cross_validation import train_test_split
import torchvision
from torchvision import transforms

class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

    # def preprocess(self, PIL_img, image_shape):
    #     rgb_mean = np.array([0.485, 0.456, 0.406])
    #     rgb_std = np.array([0.229, 0.224, 0.225])
    #     process = torchvision.transforms.Compose([
    #         torchvision.transforms.Resize(image_shape),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std),
    #         torchvision.transforms.ToPILImage(),
    #         ]
    #     )
    #     return process(PIL_img) 
        # return process(PIL_img).unsqueeze(dim = 0) # (batch_size, 3, H, W)

    # def loadToMem(self, dataPath):
    #     print("begin loading training dataset to memory")
    #     datas = {}
    #     idx = 0
    #     alphaPath = os.listdir(dataPath)[0]
    #     betaPath = os.listdir(dataPath)[1]
    #     for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
    #         datas[idx] = []
    #         imagefilePath = os.path.join(dataPath, alphaPath, charPath)
    #         datas[idx].append(self.preprocess(Image.open(imagefilePath), 105))
    #         vidoefilePath = os.path.join(dataPath, betaPath, charPath)
    #         datas[idx].append(self.preprocess(Image.open(vidoefilePath), 105))
    #         idx += 1
    #     print("finish loading training dataset to memory")
    #     return datas, idx

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx = 0
        alphaPath = os.listdir(dataPath)[0]
        betaPath = os.listdir(dataPath)[1]
        for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
            datas[idx] = []
            imagefilePath = os.path.join(dataPath, alphaPath, charPath)
            datas[idx].append((Image.open(imagefilePath)).resize((105, 105)))
            vidoefilePath = os.path.join(dataPath, betaPath, charPath)
            datas[idx].append((Image.open(vidoefilePath)).resize((105, 105)))
            idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=200):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)
        
    # def loadToMem(self, dataPath):
    #     print("begin loading test dataset to memory")
    #     datas = {}
    #     idx = 0
    #     for alphaPath in os.listdir(dataPath):
    #         for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
    #             datas[idx] = []
    #             for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
    #                 filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
    #                 datas[idx].append(Image.open(filePath).convert('L'))
    #             idx += 1
    #     print("finish loading test dataset to memory")
    #     return datas, idx

    def loadToMem(self, dataPath):
        print("begin loading testing dataset to memory")
        datas = {}
        idx = 0
        alphaPath = os.listdir(dataPath)[0]
        betaPath = os.listdir(dataPath)[1]
        for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
            datas[idx] = []
            imagefilePath = os.path.join(dataPath, alphaPath, charPath)
            datas[idx].append((Image.open(imagefilePath)).resize((105, 105)))
            vidoefilePath = os.path.join(dataPath, betaPath, charPath)
            datas[idx].append((Image.open(vidoefilePath)).resize((105, 105)))
            idx += 1
        print("finish loading testing dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2

# test
if __name__=='__main__':

    omniglotTrain = OmniglotTrain('.images_background', 30000*8)
    print(omniglotTrain)
