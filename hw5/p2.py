import matplotlib
matplotlib.use('Agg')
import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
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
#from HW5_data.reader import readShortVideo
#from HW5_data.reader import getVideoList
from p1 import training_model
from util import *
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

debug = 0
read_feature = 0
load_frame_data = 0
read_valid_txt = 0
batch_size = 32
frame_num = 16
test = 1
n_class = 11
debug_num = 10
dropout_gate = 0.0
dropout_last = 0.3
lstm_layer = 1
hidden_size = 128
learning_rate = 0.0001
#hidden_feature = []
#cnn_features_tsne = []
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
        output = output[:, -1, :]
        return output, hidden

    def forward(self, inputs, hidden=None):
        output, hidden = self.step(inputs, hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output, hidden

def training(data_loader, valid_dataloader, model, loss_filename, output_filename):
    print("start training")
    model.apply(weights_init)

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
    hidden_feature = np.zeros((1, hidden_size))
    cnn_features_tsne = np.zeros((1, 2048))
    cnt = 0
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
        hidden = np.array(hidden[0][1].data)
        cnn_feature = np.mean(np.array(cnn_feature.data), axis=1)
        if cnt == 0:
            hidden_feature = hidden
            cnn_features_tsne = cnn_feature
        else:
            hidden_feature = np.concatenate((hidden_feature, hidden), axis=0)
            cnn_features_tsne = np.concatenate((cnn_features_tsne, cnn_feature), axis=0)
        predict_label = np.array(predict_label.data)
        true_label = np.array(true_label.data)
        correct += compute_correct(predict_label, true_label)
        cnt += predict_label.shape[0]
        preds_ = np.argmax(predict_label, 1)
        for i in range(len(preds_)):
            file.write(str(preds_[i]))
            file.write('\n')

    file.close()

    print("test score: " + str(float(correct) / float(cnt)))
    return cnn_features_tsne, hidden_feature

def get_feature(data_loader, model, csvpath, output_filename):
    print("get feature...")
    model.eval()
    features = np.zeros((1, frame_num, 2048))
    for i, data in enumerate(data_loader):
        if i % 100 == 0:
            print(i)
        img = data[0].type(torch.FloatTensor)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        outputs = model.output_feature(img)
        outputs = outputs.data.cpu().numpy()
        outputs = np.reshape(outputs, (-1, frame_num, 2048))
        if i == 0:
            features = outputs
        else:
            features = np.concatenate((features, outputs), axis=0)
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]
    
    one_hot_labels = []
    for i in range(len(labels)):
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
    f = h5py.File(output_filename, 'w')
    f.create_dataset('features', data=features)

    return dataloader

def read_feature_from_file(csvpath, filename):
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

def calculate_tsne(csvpath, rnn_model, cnn_features_tsne, hidden_feature):
    print("calculate tsne...")
    color = []
    video_list = getVideoList(csvpath)
    for i, label in enumerate(video_list["Action_labels"]):
        color.append(label)
        if debug == 1 and i > 10:
            break;
    color = np.array(color)
    #CNN
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(cnn_features_tsne)
    plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE")
    plt.savefig('./cnn_tsne.jpg')
    plt.close()
    #RNN
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(hidden_feature)
    plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE")
    plt.savefig('./rnn_tsne.jpg')
    plt.close()

if __name__ == '__main__':

    torch.manual_seed(999)
    np.random.seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    #train_folder = "./HW5_data/TrimmedVideos/video/train/"
    valid_folder = sys.argv[1]#"./HW5_data/TrimmedVideos/video/valid/"
    #train_csvpath = "./HW5_data/TrimmedVideos/label/gt_train.csv"
    valid_csvpath = sys.argv[2]#"./HW5_data/TrimmedVideos/label/gt_valid.csv"
    output_folder = sys.argv[3]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, "./p2_result.txt")
    #train_feature_txt = "./train_feature.h5"
    valid_feature_txt = "./error_feature.h5"
    #train_output_frame_file = "./frames_train.h5"
    valid_output_frame_file = "./error_frame.h5"
    if test == 0:
        if read_feature == 1:
            print("read feature...")
            train_features = read_feature_from_file(train_csvpath, train_feature_txt)
            valid_features = read_feature_from_file(valid_csvpath, valid_feature_txt)
 
        else:
            print("produce feature...")
            train_dataset = extractFrames(train_folder, train_csvpath,load_frame_data,
                                          train_output_frame_file, debug, frame_num)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            valid_dataset = extractFrames(valid_folder, valid_csvpath, load_frame_data,
                                          valid_output_frame_file, debug, frame_num)
            valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            print("load p1 model...")
            model_p1 = training_model()
            model_p1.load_state_dict(torch.load('./p1.pth'))
            
            if torch.cuda.is_available():
                model_p1 = model_p1.cuda()
            train_features = get_feature(train_dataloader, model_p1, train_csvpath,
                                         train_feature_txt)
            valid_features = get_feature(valid_dataloader, model_p1, valid_csvpath,
                                         valid_feature_txt)
        print("construct RNN model...")
        if torch.cuda.is_available():
            model_RNN = RNN_model(hidden_size).cuda()
        else:
            model_RNN = RNN_model(hidden_size).cpu()
        model_RNN = training(train_features, valid_features, model_RNN,
                             "./p2_loss.jpg", output_filename)
        testing(valid_features, model_RNN, output_filename)
        #calculate_acc_from_txt(valid_csvpath, output_filename)
        
    else: 
        print("start testing...")
        valid_dataset = extractFrames(valid_folder, valid_csvpath, load_frame_data,
                                      valid_output_frame_file, debug, frame_num)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        print("load p1 model...")
        model_p1 = training_model()
        model_p1.load_state_dict(torch.load('./p1.pth'))
        
        if torch.cuda.is_available():
            model_p1 = model_p1.cuda()
        valid_features = get_feature(valid_dataloader, model_p1, valid_csvpath,
                                     valid_feature_txt)
        
        if torch.cuda.is_available():
            model_RNN = RNN_model(hidden_size).cuda()
        else:
            model_RNN = RNN_model(hidden_size).cpu()
        model_RNN.load_state_dict(torch.load('./p2.pth'))
        cnn_features_tsne, hidden_feature = testing(valid_features, model_RNN, output_filename)
        calculate_acc_from_txt(valid_csvpath, output_filename)
        #calculate_tsne(valid_csvpath, model_RNN, cnn_features_tsne, hidden_feature)
