import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

debug_num = 10
n_class = 11
batch_size = 4
def extractFrames(folder, csvpath, load, train, debug = 0):
    print("extract frames...")
    file_list = []
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
            if debug == 1 and i >= debug_num - 1:
                break

    if train == "train":
        if load == 0:
            try:
                os.remove("./frames.txt")
                os.remove("./labels.txt")
            except OSError:
                pass
            with open("./frames.txt", "wb") as fp:   #Pickling
                pickle.dump(frames, fp)
            with open("./labels.txt", "wb") as fp:   #Pickling
                pickle.dump(labels, fp)
        elif load == 1:
            with open("./frames.txt", "rb") as fp:   # Unpickling
                frames = pickle.load(fp)
            with open("./labels.txt", "rb") as fp:   # Unpickling
                labels = pickle.load(fp)
    if debug == 1:
        data = [(frames[i], labels[i]) for i in range(debug_num * batch_size)]
    else:
        data = [(frames[i], labels[i]) for i in range(len(frames))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def extractFrames2(folder, csvpath, load, train, debug = 0):
    print("extract frames...")
    file_list = []
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
            if debug == 1 and i >= debug_num - 1:
                break

    if train == "train":
        if load == 0:
            try:
                os.remove("./frames.txt")
                os.remove("./labels.txt")
            except OSError:
                pass
            with open("./frames.txt", "wb") as fp:   #Pickling
                pickle.dump(frames, fp)
            with open("./labels.txt", "wb") as fp:   #Pickling
                pickle.dump(labels, fp)
        elif load == 1:
            with open("./frames.txt", "rb") as fp:   # Unpickling
                frames = pickle.load(fp)
            with open("./labels.txt", "rb") as fp:   # Unpickling
                labels = pickle.load(fp)
    if debug == 1:
        data = [(frames[i], labels[i]) for i in range(debug_num * batch_size)]
        print(len(data))
    else:
        data = [(frames[i], labels[i]) for i in range(len(frames))]
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    return dataloader


def compute_correct(preds, labels):
    correct = 0
    preds_ = np.argmax(preds, 1)
    labels_ = np.argmax(labels, 1)
    for i in range(len(preds_)):
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


def plot_loss(all_loss, filename):

    #all_loss = butter_lowpass_filter(all_loss, 3.667, 100.0, 6)
    
    fig=plt.figure(figsize=(10, 10))
    t = np.arange(0.0, len(all_loss), 1.0)
    line, = plt.plot(t, all_loss, lw=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss vs epochs')

    plt.savefig(filename)
    plt.close()

def calculate_acc_from_txt(csvpath, output_filename):
    print("calculate acc from txt")
    labels = []
    predict = []
    correct = 0
    video_list = getVideoList(csvpath)
    labels = video_list["Action_labels"]
    file = open(output_filename, "r")
    for line in file:
        if line == '\n':
            continue
        predict.append(int(line[:-1]))
    print("len of true labels: " + str(len(labels)))
    print("len of predict labels: " + str(len(predict)))
    for i in range(len(predict)):
        if int(labels[i]) == int(predict[i]):
            correct += 1
    file.close()
    print("acc score: " + str(float(correct) / len(predict)))


