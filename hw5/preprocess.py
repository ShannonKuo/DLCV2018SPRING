#import matplotlib
#matplotlib.use('Agg')
import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F
import scipy.misc
import os
import sys
import numpy as np
import cv2
import csv
import skvideo.io
import skimage.transform
import collections
from HW5_data.reader import readShortVideo
from HW5_data.reader import getVideoList
#import matplotlib.pyplot as plt
#from scipy.signal import butter, lfilter, freqz

debug = 1
batch_size = 32
learning_rate = 1e-5
n_class = 11
if debug == 1:
    num_epochs = 1
else:
    num_epochs = 10

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

def extractFrames(folder, csvpath):
    print("extract frames...")
    file_list = []
    video_frame_num = []
    frames = []
    labels = []
    video_list = getVideoList(csvpath)
    cnt = 0
    for i in range(len(video_list["Video_name"])):
        if debug == 1 and i > 2:
            break
        frame = readShortVideo(folder, video_list["Video_category"][i],
                                video_list["Video_name"][i])
        for j in range(len(frame)):
            frames.append(np.moveaxis(frame[j], -1, 0))
            label = np.zeros(n_class)
            label[int(video_list["Action_labels"][i])] = 1
            labels.append(label)
            cnt += 1
        video_frame_num.append(len(frame))

    data = [(frames[i], labels[i]) for i in range(len(frames))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader, video_frame_num

class training_model(nn.Module):
    def __init__(self):
        super(training_model, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained=True)
        self.pretrained.fc = nn.Linear(16 * 32 * 32, 2048)
        
        self.fcn = nn.Linear(2048, n_class)
        self.softmax = nn.Softmax()

    def output_feature(self, x):
        x = self.pretrained(x)
        x = torch.squeeze(x, 1)
        return(x)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.fcn(x)
        x = self.softmax(x)
        x = torch.squeeze(x, 1)
        return x


def training(data_loader):
    print("start training")
    if torch.cuda.is_available():
        model = training_model().cuda()
    else:
        model = training_model().cpu()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

    all_loss = []
    for epoch in range(num_epochs):
        idx = 0
        train_loss = 0.0
        for data in data_loader:
            img = data[0].type(torch.FloatTensor)
            true_label = data[1].type(torch.FloatTensor)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                true_label = Variable(true_label).cuda()
            else:
                img = Variable(img).cpu()
                true_label = Variable(true_label).cpu()
            #print(type(img))
            #img_show = to_img(img.cpu().data)
            # ===================forward=====================
            predict_label = model(img)
            loss = nn.BCELoss()(predict_label, true_label)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
        torch.save(model.state_dict(), './p1.pth')

    return model




def get_feature(data_loader, model, video_frame_num):
    print("get feature...")
    features_one_video = []
    features = []
    video_id = 0
    for i, data in enumerate(data_loader):
        img = data[0].type(torch.FloatTensor)
        true_label = data[1].type(torch.FloatTensor)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            true_label = Variable(true_label).cuda()
        else:
            img = Variable(img).cpu()
            true_label = Variable(true_label).cpu()
        outputs = model.output_feature(img)
        
        for j, output in enumerate(outputs):
            if video_frame_num[video_id] > 1:
                features_one_video.append(output)
                video_frame_num[video_id] -= 1

            else:
                array_feature = np.array(features_one_video)
                array_feature = np.reshape(array_feature, (-1, 2048))
                print(array_feature.shape)
                avg_feature = np.mean(array_feature, axis = 0)
                print(avg_feature.shape)
                features.append(avg_feature)
                video_id += 1
                features_one_video = []
    return features

if __name__ == '__main__':
    folder = "./HW5_data/TrimmedVideos/video/train/"
    csvpath = "./HW5_data/TrimmedVideos/label/gt_train.csv"
    train_dataloader, video_frame_num = extractFrames(folder, csvpath)
    #valid_dataloader = extractFrames("./HW5_data/TrimmedVideos/video/valid/")
    model = training(train_dataloader)
    train_features = get_feature(train_dataloader, model, video_frame_num)
    
    #valid_features = get_feature(valid_dataloader, model)
    
