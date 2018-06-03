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
import h5py
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
read_feature = 1
load_frame_data = 1
read_valid_txt = 0
batch_size = 16
test = 0
n_class = 11
debug_num = 10
dropout_gate = float(sys.argv[1])
dropout_last = float(sys.argv[2])
lstm_layer = int(sys.argv[3])
hidden_size = int(sys.argv[4])
learning_rate = float(sys.argv[5])

if debug == 1:
    num_epochs = 1
else:
    num_epochs = 60

def weights_init(m):
    for name, param in m.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 1.0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=1)

class RNN_model(nn.Module):
    def __init__(self, hidden_size):
        super(RNN_model, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(2048, hidden_size, lstm_layer, dropout=dropout_gate, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_last)
        self.out = nn.Linear(hidden_size * 2, 11)
        self.softmax = nn.Softmax()

    def step(self, input, hidden=None):
        input = input.view(len(input), -1).unsqueeze(1)

        h0 = torch.zeros(lstm_layer*2, input.size(1), self.hidden_size) # 2 for bidirection 
        c0 = torch.zeros(lstm_layer*2, input.size(1), self.hidden_size)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        output, hidden = self.rnn(input, (h0, c0))
        #output, hidden = self.rnn(input, hidden)
        return output, hidden

    def forward(self, inputs, hidden=None, steps=0):
        if steps == 0: steps = len(inputs)
        output, hidden = self.step(inputs, hidden)
        output = self.dropout(output)
        output = self.out(output).view(len(inputs), -1)
        output = output[-1]
        output = self.softmax(output)
        output = output.view(1, -1)
        return output, hidden

def training(data_loader, valid_dataloader, model, loss_filename, output_filename):
    print("start training")
    model.apply(weights_init)
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

def get_feature(data_loader, model, csvpath, output_filename, mode):
    print("get feature...")
    """if debug == 1:
        features = np.zeros((batch_size * 10, 2048))
    elif mode == "train":
        features = np.zeros((batch_size * 3236, 2048))
    elif mode == "valid":
        features = np.zeros((batch_size * 517, 2048))
    elif model == "test":
        features = np.zeros((batch_size * 398, 2048))
    """
    features = np.zeros((1, 2048))
    cnt = 0
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print(i)
        img = data[0].type(torch.FloatTensor)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        outputs = model.output_feature(img)
        outputs = outputs.data.cpu().numpy()
        for j in range(outputs.shape[0]):
            output = np.reshape(outputs[j], (-1, 2048))
            if cnt == 0:
                features = output
            else:
                features = np.concatenate((features, output), axis=0)
            cnt += 1
    print(features.shape)    
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
    
    try:
        os.remove(output_filename)
    except OSError:
        pass
    print("start produce feature .h5")
    #with open(output_filename, "wb") as fp:   #Pickling
    #    pickle.dump(features, fp)
    f = h5py.File(output_filename, 'w')
    f.create_dataset('features', data=features)

    return dataloader

def read_feature_from_file(csvpath, filename):
    #features = []
    #with open(filename, "rb") as fp:   # Unpickling
    #    features = pickle.load(fp)
    f = h5py.File(filename, 'r')
    features = f['features'][:]
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]

    one_hot_labels = []
    for i in range(len(labels)):
        for j in range(batch_size):
            label = np.zeros(n_class)
            label[int(video_list["Action_labels"][i])] = 1
            one_hot_labels.append(label)
    print("len of feature: " + str(features.shape[0]))
    print("feature size: " + str(features[0].shape))
    data = [(features[i], one_hot_labels[i]) for i in range(len(features))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    train_feature_txt = "./train_feature.h5"
    valid_feature_txt = "./valid_feature.h5"
    if test == 0:
        if read_feature == 1:
            print("read feature...")
            train_features = read_feature_from_file(train_csvpath, train_feature_txt)
            valid_features = read_feature_from_file(valid_csvpath, valid_feature_txt)
 
        else:
            print("produce feature...")
            train_dataloader = extractFrames2(train_folder, train_csvpath, load_frame_data, "train", debug, batch_size)
            valid_dataloader = extractFrames2(valid_folder, valid_csvpath, load_frame_data, "valid", debug, batch_size)
            print("load p1 model...")
            model_p1 = training_model()
            model_p1.load_state_dict(torch.load('./p1.pth'))
            
            if torch.cuda.is_available:
                model_p1 = model_p1.cuda()
            train_features = get_feature(train_dataloader, model_p1, train_csvpath, train_feature_txt, "train")
            valid_features = get_feature(valid_dataloader, model_p1, valid_csvpath, valid_feature_txt, "valid")
        print("construct RNN model...")
        if torch.cuda.is_available():
            model_RNN = RNN_model(hidden_size).cuda()
        else:
            model_RNN = RNN_model(hidden_size).cpu()
        print(count_parameters(model_RNN))
        print(model_RNN)
        model_RNN = training(train_features, valid_features, model_RNN, "./p2_loss.jpg", output_filename)
        testing(valid_features, model_RNN, output_filename)
        calculate_acc_from_txt(valid_csvpath, output_filename)
        
    else: 
        model_RNN = RNN_model.model()
        model_RNN.load_state_dict(torch.load('./p2.pth'))
        testing(valid_features, model_RNN, output_filename)
        calculate_acc_from_txt(valid_csvpath, output_filename)
