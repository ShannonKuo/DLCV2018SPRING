import matplotlib
matplotlib.use('Agg')
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
import pickle
from HW5_data.reader import readShortVideo
from HW5_data.reader import getVideoList
from util import *
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

debug = 1
read_valid_txt = 0
batch_size = 4
learning_rate = 1e-5
n_class = 11
debug_num = 10
if debug == 1:
    num_epochs = 1
else:
    num_epochs = 30


class training_model(nn.Module):
    def __init__(self):
        super(training_model, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained=True)
        self.pretrained.fc = nn.Linear(16 * 32 * 32, 2048)
        
        self.fcn = nn.Linear(2048, n_class)
        self.softmax = nn.Softmax()

    def output_feature(self, x):
        x = self.pretrained(x)
        avg_feature = np.mean(np.array(x.data), axis = 0)
        avg_feature = np.reshape(avg_feature, (1, 2048))
        avg_feature = torch.from_numpy(avg_feature)
        avg_feature = torch.squeeze(avg_feature, 1)
        if torch.cuda.is_available():
            avg_feature = Variable(avg_feature).cuda()
        else:
            avg_feature = Variable(avg_feature).cpu()

        return(avg_feature)

    def forward(self, x):
        x = self.pretrained(x)
        avg_feature = np.mean(np.array(x.data), axis = 0)
        avg_feature = np.reshape(avg_feature, (1, 2048))
        avg_feature = torch.from_numpy(avg_feature)
        avg_feature = torch.squeeze(avg_feature, 1)
        if torch.cuda.is_available():
            avg_feature = Variable(avg_feature).cuda()
        else:
            avg_feature = Variable(avg_feature).cpu()

        z = self.fcn(avg_feature)
        z = self.softmax(z)
        z = torch.squeeze(z, 1)
        return z

def training(data_loader, valid_dataloader, loss_filename):
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
            true_label = true_label[0].view(1, n_class)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                true_label = Variable(true_label).cuda()
            else:
                img = Variable(img).cpu()
                true_label = Variable(true_label).cpu()
            # ===================forward=====================
            predict_label = model(img)
            loss = nn.BCELoss()(predict_label, true_label)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
        torch.save(model.state_dict(), './p1.pth')
        all_loss.append(loss.item())
        testing(valid_dataloader, model)

    plot_loss(all_loss, loss_filename)
    return model

def testing(data_loader, model):
    cnt = 0
    correct = 0
    save_filename = './p1_valid.txt'
    try:
        os.remove(save_filename)
    except OSError:
        pass
    file = open(save_filename, "a+")

    for data in data_loader:
        img = data[0].type(torch.FloatTensor)
        true_label = data[1].type(torch.FloatTensor)
        true_label = true_label[0].view(1, n_class)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            true_label = Variable(true_label).cuda()
        else:
            img = Variable(img).cpu()
            true_label = Variable(true_label).cpu()
        # ===================forward=====================
        predict_label = model(img)
        predict_label = np.array(predict_label.data)
        true_label = np.array(true_label.data)
        correct += compute_correct(predict_label, true_label)
        cnt += predict_label.shape[0]
        preds_ = np.argmax(predict_label, 1)
        for i in range(len(preds_)):
            file.write(str(preds_[i]))
            file.write('\n')

    file.write('\n')
    file.close()

    print("test score: " + str(float(correct) / float(cnt)))


if __name__ == '__main__':

    torch.manual_seed(999)
    np.random.seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

        
    train_folder = "./HW5_data/TrimmedVideos/video/train/"
    valid_folder = "./HW5_data/TrimmedVideos/video/valid/"
    train_csvpath = "./HW5_data/TrimmedVideos/label/gt_train.csv"
    valid_csvpath = "./HW5_data/TrimmedVideos/label/gt_valid.csv"

    if read_valid_txt == 1:
        calculate_acc_from_txt(valid_csvpath)
    else:
        train_dataloader = extractFrames(train_folder, train_csvpath, 0, "train", debug)
        valid_dataloader = extractFrames(valid_folder, valid_csvpath, 0, "valid", debug)
        model = training(train_dataloader, valid_dataloader, "./loss.jpg")
        testing(valid_dataloader, model)
        calculate_acc_from_txt(valid_csvpath, "./p1_valid.txt")
        """train_features = get_feature(train_dataloader, model)
        valid_features = get_feature(valid_dataloader, model)
        
        try:
            os.remove("./train_features.txt")
            os.remove("./valid_features.txt")
        except OSError:
            pass
     
        with open("./train_features.txt", "wb") as fp:   #Pickling
            pickle.dump(train_features, fp)
        with open("./valid_features.txt", "wb") as fp:   #Pickling
            pickle.dump(valid_features, fp)
        """
