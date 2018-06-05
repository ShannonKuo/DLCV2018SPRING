import matplotlib
matplotlib.use('Agg')
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
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

debug = 1
read_feature = 0
load_frame_data = 0
read_valid_txt = 0
batch_size = 32
frame_rate = 64
test = 0
n_class = 11
debug_num = 3
dropout_gate = 0.0
dropout_last = 0.3
lstm_layer = 1
hidden_size = 128
learning_rate = 0.0001

if debug == 1:
    num_epochs = 1
else:
    num_epochs = 25

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

        self.rnn = nn.LSTM(2048, hidden_size, lstm_layer, dropout=dropout_gate, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_last)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 11)
        self.softmax = nn.Softmax()

    def step(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def forward(self, inputs, hidden=None):
        pack = torch.nn.utils.rnn.pack_padded_sequence(inputs, 500, batch_first=True)
        output, hidden = self.step(pack, hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output, hidden

def training(data_loader, valid_dataloader, model, loss_filename, output_filename):
    print("start training")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    all_loss = []
    for epoch in range(num_epochs):
        model.train()
        idx = 0
        train_loss = 0.0
        for i, data in enumerate(data_loader):
            cnn_feature = data[0].type(torch.FloatTensor)
            true_label = data[1].type(torch.FloatTensor)
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
        plot_loss(all_loss, loss_filename)
        testing(valid_dataloader, model, output_filename)

    return model

def testing(data_loader, model, save_filename):
    cnt = 0
    correct = 0
    model.eval()
    try:
        os.remove(save_filename)
    except OSError:
        pass
    file = open(save_filename, "a+")

    for data in data_loader:
        cnn_feature = data[0].type(torch.FloatTensor)
        true_label = data[1].type(torch.FloatTensor)
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

def get_feature(data_loader, model, output_filename, img_folder, label_folder):
    print("get feature...")
    model.eval()
    features = np.zeros((1, frame_rate, 2048))
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print(i)
        img = data[0].type(torch.FloatTensor)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        outputs = model.output_feature(img)
        outputs = outputs.data.cpu().numpy()
        outputs = np.reshape(outputs, (-1, frame_rate, 2048))
        if i == 0:
            features = outputs
        else:
            features = np.concatenate((features, outputs), axis=0)
    all_labels = read_labels_p3(img_folder, label_folder, debug, frame_rate)
    data = [(features[i], all_labels[i]) for i in range(len(features))]
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

def read_feature_from_file(filename, label_folder):
    f = h5py.File(filename, 'r')
    features = f['features'][:]
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]

    one_hot_labels = []
    for i in range(len(labels)):
        label = np.zeros(n_class)
        label[int(video_list["Action_labels"][i])] = 1
        one_hot_labels.append(label)
    print("feature size: " + str(features.shape))
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

    train_img_folder = "./HW5_data/FullLengthVideos/videos/train/"
    train_label_folder = "./HW5_data/FullLengthVideos/labels/train/"
    valid_img_folder = "./HW5_data/FullLengthVideos/videos/valid/"
    valid_label_folder = "./HW5_data/FullLengthVideos/labels/valid/"
    output_filename = "./p3_result.txt"
    train_feature_txt = "./train_feature_p3.h5"
    valid_feature_txt = "./valid_feature_p3.h5"

    if test == 0:
        if read_feature == 1:
            print("read feature...")
            train_features = read_feature_from_file(train_feature_txt)
            valid_features = read_feature_from_file(valid_feature_txt)
 
        else:
            print("produce feature...")
            train_dataset = extractFrames_p3(train_img_folder, train_label_folder,
                                           debug, frame_rate)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            valid_dataset = extractFrames_p3(valid_img_folder, valid_label_folder,
                                          debug, frame_rate)
            valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            print("load p1 model...")
            model_p1 = training_model()
            if torch.cuda.is_available():
                model_p1 = model_p1.cuda()
            else:
                model_p1 = model_p1.cpu()
            model_p1.load_state_dict(torch.load('./p1.pth'))
            
            train_features = get_feature(train_dataloader, model_p1,
                                         train_feature_txt, train_img_folder, train_label_folder)
            valid_features = get_feature(valid_dataloader, model_p1,
                                         valid_feature_txt, valid_img_folder, valid_label_folder)
        print("construct RNN model...")
        if torch.cuda.is_available():
            model_RNN = RNN_model(hidden_size).cuda()
        else:
            model_RNN = RNN_model(hidden_size).cpu()
        model_RNN.load_state_dict(torch.load('./p2.pth'))

        model_RNN = training(train_features, valid_features, model_RNN,
                             "./p3_loss.jpg", output_filename)
        testing(valid_features, model_RNN, output_filename)
        calculate_acc_from_txt(valid_csvpath, output_filename)
        
    else: 
        print("start testing...")
        valid_dataset = extractFrames_p3(valid_img_folder, valid_label_folder,
                                       0, frame_rate)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        print("load p1 model...")
        model_p1 = training_model()
        
        if torch.cuda.is_available():
            model_p1 = model_p1.cuda()
        model_p1.load_state_dict(torch.load('./p1.pth'))
        valid_features = get_feature(valid_dataloader, model_p1,
                                     valid_feature_txt, valid_img_folder, valid_label_folder)

        if torch.cuda.is_available():
            model_RNN = RNN_model(hidden_size).cuda()
        else:
            model_RNN = RNN_model(hidden_size).cpu()
        model_RNN.load_state_dict(torch.load('./p2.pth'))
        testing(valid_features, model_RNN, output_filename)
        calculate_acc_from_txt(valid_csvpath, output_filename)
