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
import csv
import h5py
import skvideo.io
import skimage.transform
import collections
import pickle
from HW5_data.reader import readShortVideo
from HW5_data.reader import getVideoList

debug_num = 10
n_class = 11
batch_size = 32

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def extractFrames(folder, csvpath, load, output_filename, debug = 0, frame_num=16, batch_size=32):
    print("extract frames...")
    labels = []
    video_list = getVideoList(csvpath)
    frames = np.zeros((batch_size, frame_num, 240, 320, 3))
    cnt = 0
    
    if (load == 0):
        for i in range(len(video_list["Video_name"])):
            frame = readShortVideo(folder, video_list["Video_category"][i],
                                    video_list["Video_name"][i], frame_num=frame_num)
            #frame = np.moveaxis(frame, -1, 1)
            frame = frame.reshape(1, frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3])
            if i == 0:
                frames = frame
            else:
                frames = np.concatenate((frames, frame), axis=0)
            if i % 100 == 0:
                print(i)
                print(frames.shape)
            if debug == 1 and i >= debug_num - 1:
                break
    frames = np.moveaxis(frames, -1, 2)
    for i in range(len(video_list["Video_name"])):
        label = np.zeros(n_class)
        label[int(video_list["Action_labels"][i])] = 1
        labels.append(label)
        
        if debug == 1 and i >= debug_num - 1:
            break

    if load == 0:
        try:
            os.remove(output_filename)
        except OSError:
            pass
        f = h5py.File(output_filename, "w")
        f.create_dataset("frames", data = frames)
    elif load == 1:
        f = h5py.File(output_filename, "r")
        frames = f['frames'][:]

    if debug == 1:
        data = [(frames[i], labels[i]) for i in range(debug_num)]
    else:
        data = [(frames[i], labels[i]) for i in range(frames.shape[0])]
    print(frames.shape)
    print(len(labels))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
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


