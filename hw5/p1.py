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

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

def extractFrames(folder, csvpath, load, train):
    print("extract frames...")
    file_list = []
    video_frame_num = []
    frames = []
    labels = []
    video_list = getVideoList(csvpath)
    cnt = 0
    if (load == 0):
        for i in range(len(video_list["Video_name"])):
            frame = readShortVideo(folder, video_list["Video_category"][i],
                                    video_list["Video_name"][i])
            for j in range(len(frame)):
                frames.append(np.moveaxis(frame[j], -1, 0))
                label = np.zeros(n_class)
                label[int(video_list["Action_labels"][i])] = 1
                labels.append(label)
                cnt += 1
            video_frame_num.append(len(frame))
            if debug == 1 and i >= debug_num - 1:
                break

    if train == "train":
        if load == 0:
            try:
                os.remove("./frames.txt")
                os.remove("./labels.txt")
                os.remove("./video_frame_num.txt")
            except OSError:
                pass
            with open("./frames.txt", "wb") as fp:   #Pickling
                pickle.dump(frames, fp)
            with open("./labels.txt", "wb") as fp:   #Pickling
                pickle.dump(labels, fp)
            with open("./video_frame_num.txt", "wb") as fp:   #Pickling
                pickle.dump(video_frame_num, fp)
        elif load == 1:
            with open("./frames.txt", "rb") as fp:   # Unpickling
                frames = pickle.load(fp)
            with open("./labels.txt", "rb") as fp:   # Unpickling
                labels = pickle.load(fp)
            with open("./video_frame_num.txt", "rb") as fp:   # Unpickling
                video_frame_num = pickle.load(fp)
    print(len(frames))
    print(len(labels))
    if debug == 1:
        data = [(frames[i], labels[i]) for i in range(debug_num * batch_size)]
    else:
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
        print(x.shape)
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

def training(data_loader, valid_dataloader):
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
            print("train predict")
            # ===================forward=====================
            predict_label = model(img)
            loss = nn.BCELoss()(predict_label, true_label)
            print(predict_label.data)
            print(true_label.data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
        torch.save(model.state_dict(), './p1.pth')
        all_loss.append(loss.data[0])
        testing(valid_dataloader, model)

    plot_loss(all_loss)
    return model

def compute_correct(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    labels_ = labels.data.max(1)[1]
    for i in range(len(preds_)):
        print(preds_[i], labels_[i])
        if preds_[i] == labels_[i]:
            correct += 1
    return correct

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_loss(all_loss):

    #all_loss = butter_lowpass_filter(all_loss, 3.667, 100.0, 6)
    
    fig=plt.figure(figsize=(10, 10))
    t = np.arange(0.0, len(all_loss), 1.0)
    line, = plt.plot(t, all_loss, lw=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss vs epochs')

    plt.savefig('./loss.jpg')
    plt.close()

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
        print("predict")
        correct += compute_correct(predict_label, true_label)
        cnt += predict_label.shape[0]
        preds_ = predict_label.data.max(1)[1]
        for i in range(len(preds_)):
            file.write(str(preds_[i]))
            file.write('\n')

    file.write('\n')
    file.close()

    print("test score: " + str(float(correct) / float(cnt)))

def calculate_acc_from_txt(csvpath):
    print("calculate acc from txt")
    labels = []
    predict = []
    correct = 0
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]
    if debug == 1:
        labels = labels[:debug_num * batch_size]
    file = open("./p1_valid.txt", "r")
    for line in file:
        if line == '\n':
            continue
        predict.append(int(line[:-1]))
    print("len of true labels: " + str(len(labels)))
    print("leb of predict lables: " + str(len(predict)))
    for i in range(len(predict)):
        if int(labels[i]) == int(predict[i]):
            correct += 1
    file.close()
    print("acc score: " + str(float(correct) / len(predict)))


def get_feature(data_loader, model, video_frame_num):
    print("get feature...")
    features = []
    for i, data in enumerate(data_loader):
        img = data[0].type(torch.FloatTensor)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = Variable(img).cpu()
        outputs = model.output_feature(img)
        features.append(outputs)    
    return features

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
        train_dataloader, train_video_frame_num = extractFrames(train_folder, train_csvpath, 0, "train")
        valid_dataloader, valid_video_frame_num = extractFrames(valid_folder, valid_csvpath, 0, "valid")
        model = training(train_dataloader, valid_dataloader)
        testing(valid_dataloader, model)
        calculate_acc_from_txt(valid_csvpath)
        train_features = get_feature(train_dataloader, model, train_video_frame_num)
        valid_features = get_feature(valid_dataloader, model, valid_video_frame_num)
        
        try:
            os.remove("./train_features.txt")
            os.remove("./valid_features.txt")
        except OSError:
            pass
     
        with open("./train_features.txt", "wb") as fp:   #Pickling
            pickle.dump(train_features, fp)
        with open("./valid_features.txt", "wb") as fp:   #Pickling
            pickle.dump(valid_features, fp)
