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
from p1 import training_model
from util import *
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

debug = 0
load_frame_data = 1
read_valid_txt = 0
batch_size = 4
learning_rate = 1e-4
n_class = 11
hidden_size = 2048
debug_num = 10
if debug == 1:
    num_epochs = 1
else:
    num_epochs = 100

class RNN_model(nn.Module):
    def __init__(self, hidden_size):
        super(RNN_model, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(hidden_size, hidden_size, 1, dropout=0.05)
        self.out = nn.Linear(hidden_size, 11)
        self.softmax = nn.Softmax()

    def step(self, input, hidden=None):
        input = input.view(1, -1).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def forward(self, inputs, hidden=None, steps=0):
        if steps == 0: steps = len(inputs)
        for i in range(steps):
            if i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
        output = self.out(output).view(1, -1)
        output = self.softmax(output)
        return output, hidden

def training(data_loader, valid_dataloader, model, loss_filename, output_filename):
    print("start training")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

    all_loss = []
    for epoch in range(num_epochs):
        idx = 0
        train_loss = 0.0
        for data in data_loader:
            cnn_feature = data[0].type(torch.FloatTensor)
            true_label = data[1].type(torch.FloatTensor)
            true_label = true_label[0].view(1, n_class)
            if torch.cuda.is_available():
                cnn_feature = Variable(cnn_feature).cuda()
                true_label = Variable(true_label).cuda()
            else:
                cnn_feature = Variable(cnn_feature).cpu()
                true_label = Variable(true_label).cpu()
            # ===================forward=====================
            predict_label, hidden = model(cnn_feature, None)
            loss = nn.BCELoss()(predict_label, true_label)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
        torch.save(model.state_dict(), './p2.pth')
        all_loss.append(loss.item())
        testing(valid_dataloader, model, output_filename)

        plot_loss(all_loss, loss_filename)
    return model

def testing(data_loader, model, save_filename):
    cnt = 0
    correct = 0
    try:
        os.remove(save_filename)
    except OSError:
        pass
    file = open(save_filename, "a+")

    for data in data_loader:
        cnn_feature = data[0].type(torch.FloatTensor)
        true_label = data[1].type(torch.FloatTensor)
        true_label = true_label[0].view(1, n_class)
        if torch.cuda.is_available():
            cnn_feature = Variable(cnn_feature).cuda()
            true_label = Variable(true_label).cuda()
        else:
            cnn_feature = Variable(cnn_feature).cpu()
            true_label = Variable(true_label).cpu()
        # ===================forward=====================
        predict_label, hidden = model(cnn_feature, None)
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

def get_feature(data_loader, model, csvpath):
    print("get feature...")
    features = []
    for i, data in enumerate(data_loader):
        img = data[0].type(torch.FloatTensor)
        #if torch.cuda.is_available():
        #    img = Variable(img).cuda()
        #else:
        img = Variable(img).cpu()
        outputs = model.output_feature(img)
        features.append(outputs.data)

    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]

    one_hot_labels = []
    for i in range(len(labels)):
        for j in range(batch_size):
            label = np.zeros(n_class)
            label[int(video_list["Action_labels"][i])] = 1
            one_hot_labels.append(label)

    data = [(features[i], one_hot_labels[i]) for i in range(len(features))]

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


if __name__ == '__main__':

    torch.manual_seed(999)
    np.random.seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    train_folder = "./HW5_data/TrimmedVideos/video/train/"
    valid_folder = "./HW5_data/TrimmedVideos/video/valid/"
    train_csvpath = "./HW5_data/TrimmedVideos/label/gt_train.csv"
    valid_csvpath = "./HW5_data/TrimmedVideos/label/gt_valid.csv"
    output_filename = "./p2_result.txt"

    train_dataloader = extractFrames2(train_folder, train_csvpath, load_frame_data, "train", debug)
    valid_dataloader = extractFrames2(valid_folder, valid_csvpath, 0, "valid", debug)
    print("load p1 model...")
    model_p1 = training_model()
    model_p1.load_state_dict(torch.load('./p1.pth'))
    train_features = get_feature(train_dataloader, model_p1, train_csvpath)
    valid_features = get_feature(valid_dataloader, model_p1, valid_csvpath)

    print("construct RNN model...")
    if torch.cuda.is_available():
        model_RNN = RNN_model(hidden_size).cuda()
    else:
        model_RNN = RNN_model(hidden_size).cpu()

    model_RNN = training(train_features, valid_features, model_RNN, "./p2_loss.jpg", output_filename)
    testing(valid_features, model_RNN, output_filename)
    calculate_acc_from_txt(valid_csvpath, output_filename)
