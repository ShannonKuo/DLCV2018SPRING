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
from HW5_data.reader import readShortVideo
from HW5_data.reader import getVideoList
from p1 import training_model
from util import *
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

debug = 0
read_feature = 0
load_frame_data = 0
read_valid_txt = 0
batch_size = 1
frame_num = 256
test = 1
mode = ""
n_class = 11
debug_num = 2
dropout_gate = 0.0
dropout_last = 0.0
lstm_layer = 1
hidden_size = 128
learning_rate = 0.0001

if debug == 1:
    num_epochs = 1
else:
    num_epochs = 50

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
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 11)
        self.softmax = nn.Softmax()

    def step(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def forward(self, inputs, hidden=None):
        output, hidden = self.step(inputs, hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = output[0]
        output = self.softmax(output)
        output = output.view(batch_size, -1, n_class)
        return output, hidden

def training(data_loader, valid_dataloader, model, loss_filename,
             valid_img_folder, output_folder):
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
            loss = nn.BCELoss()(predict_label[0], true_label[0])
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
        torch.save(model.state_dict(), './p3.pth')
        all_loss.append(loss.item())
        plot_loss(all_loss, loss_filename)
        testing(valid_dataloader, model, valid_img_folder, output_folder)

    return model

def testing(data_loader, model, img_folder, output_folder):
    model.eval()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dirNames = [subfolder for subfolder in os.listdir(img_folder) if not subfolder.startswith('.')]
    dir_id = 0

    for data in data_loader:
        cnn_feature = data[0].type(torch.FloatTensor)
        #true_label = data[1].type(torch.FloatTensor)
        length = data[1]
        #true_label = true_label[:, 0: length, :]
        if torch.cuda.is_available():
            cnn_feature = Variable(cnn_feature).cuda()
            #true_label = Variable(true_label).cuda()
        else:
            cnn_feature = Variable(cnn_feature).cpu()
            #true_label = Variable(true_label).cpu()
        # ===================forward=====================
        predict_label, hidden = model(cnn_feature, None)
        predict_label = np.array(predict_label.data)
        #true_label = np.array(true_label.data)
        for j in range(predict_label.shape[0]):
            #correct += compute_correct(predict_label[j][:length], true_label[j][:length])
            #cnt += length
            preds_ = np.argmax(predict_label[j], 1)
            save_filename = os.path.join(output_folder, dirNames[dir_id]) + ".txt"
            try:
                os.remove(save_filename)
            except OSError:
                pass
            file = open(save_filename, "a+")
            for i in range(length.data[j]):
                file.write(str(preds_[i]))
                file.write('\n')
            file.close()
            dir_id += 1

    #print("test score: " + str(float(correct) / float(cnt)))

def get_feature(data_loader, model, output_filename, img_folder, label_folder):
    print("get feature...")
    model.eval()
    features = np.zeros((1, frame_num, 2048))
    length = []
    for i, data in enumerate(data_loader):
        img = data[0]
        frame_length = img.shape[1]
        group_size = int(frame_length / 30)
        length.append(int(data[1]))
        for j in range(30):
            if j == 29:
                input = img[:, group_size * j:]
            else:
                input = img[:, group_size * j: group_size * (j+1)]
            if torch.cuda.is_available():
                input = Variable(input.type(torch.FloatTensor)).cuda()
            output = model.output_feature(input)
            output = output.data.cpu().numpy()
            if j == 0:
                outputs = output
            else:
                outputs = np.concatenate((outputs, output), axis=1)
        if i == 0:
            features = outputs
        else:
            features = np.concatenate((features, outputs), axis=0)
    if mode == "train":
        all_labels = read_labels_p3(img_folder, label_folder, debug, frame_num, mode)
        data = [(features[i], all_labels[i], length[i]) for i in range(len(features))]
    else:
        data = [(features[i], length[i]) for i in range(len(features))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    try:
        os.remove(output_filename)
    except OSError:
        pass
    print("start produce feature .h5")
    f = h5py.File(output_filename, 'w')
    f.create_dataset('features', data=features)

    return dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    torch.manual_seed(999)
    np.random.seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    #train_img_folder = "./HW5_data/FullLengthVideos/videos/train/"
    #train_label_folder = "./HW5_data/FullLengthVideos/labels/train/"
    valid_img_folder = sys.argv[1]#"./HW5_data/FullLengthVideos/videos/valid/"
    valid_label_folder = "./error/"
    #train_feature_txt = "./train_feature_p3.h5"
    valid_feature_txt = "./error_p3.h5"
    output_label_folder = sys.argv[2]#"./p3_output/"
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    if test == 0:
        if read_feature == 1:
            print("read feature...")
            mode = "train"
            train_features = read_feature_from_file(train_feature_txt, 
                                                    train_img_folder, train_label_folder)
            mode = "test"
            valid_features = read_feature_from_file(valid_feature_txt,
                                                    valid_img_folder, valid_label_folder)
 
        else:
            print("produce feature...")
            mode = "train"
            train_dataset = extractFrames_p3(train_img_folder, train_label_folder,
                                           debug, frame_num, mode)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

            mode = "test"
            valid_dataset = extractFrames_p3(valid_img_folder, valid_label_folder,
                                          debug, frame_num, mode)
            valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            print("load p1 model...")
            model_p1 = training_model()
            if torch.cuda.is_available():
                model_p1 = model_p1.cuda()
            else:
                model_p1 = model_p1.cpu()
            model_p1.load_state_dict(torch.load('./p1.pth'))
            mode = "train"
            train_features = get_feature(train_dataloader, model_p1,
                                         train_feature_txt, train_img_folder, train_label_folder)
            mode = "test"
            valid_features = get_feature(valid_dataloader, model_p1,
                                         valid_feature_txt, valid_img_folder, valid_label_folder)
        print("construct RNN model...")
        if torch.cuda.is_available():
            model_RNN = RNN_model(hidden_size).cuda()
        else:
            model_RNN = RNN_model(hidden_size).cpu()

        model_RNN = training(train_features, valid_features, model_RNN,
                             "./p3_loss.jpg", valid_img_folder, output_label_folder)
        testing(valid_features, model_RNN, valid_img_folder, output_label_folder)
        calculate_acc_from_txt_p3(valid_label_folder, output_label_folder)
        
    else: 
        print("start testing...")
        mode = "test"
        valid_dataset = extractFrames_p3(valid_img_folder, valid_label_folder,
                                       0, frame_num, mode)
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
        model_RNN.load_state_dict(torch.load('./p3.pth'))
        testing(valid_features, model_RNN, valid_img_folder, output_label_folder)
        valid_label_folder = "./HW5_data/FullLengthVideos/labels/valid/"
        calculate_acc_from_txt_p3(valid_label_folder, output_label_folder)
