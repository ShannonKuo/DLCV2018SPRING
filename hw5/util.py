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
from skimage import io
import skimage.transform
import collections
import pickle
#from HW5_data.reader import readShortVideo
#from HW5_data.reader import getVideoList

debug_num = 3
max_frame_num = 64
n_class = 11

def extractFrames(folder, csvpath, load, output_filename, debug = 0, frame_num=16):
    print("extract frames...")
    labels = []
    video_list = getVideoList(csvpath)
    frames = np.zeros((1, frame_num, 240, 320, 3))
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
            if debug == 1 and i >= debug_num - 1:
                break
        frames = np.moveaxis(frames, -1, 2)
    for i in range(len(video_list["Video_name"])):
        label = np.zeros(n_class)
        label[int(video_list["Action_labels"][i])] = 1
        labels.append(label.astype(np.float))
        
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
        print("read frames")
        f = h5py.File(output_filename, "r")
        frames = f['frames'][:]

    if debug == 1:
        data = [(frames[i], labels[i]) for i in range(debug_num)]
    else:
        data = [(frames[i], labels[i]) for i in range(frames.shape[0])]
    return data

def extractFrames_p3(img_folder, label_folder, debug = 0, frame_rate=64):
    print("extract frames...")
    video_num = 0
    videos = np.zeros((1, max_frame_num, 240, 320, 3))
    #dirPath, dirNames, filenames = os.walk(img_folder)
    dirNames = [os.path.join(img_folder, subfolder) for subfolder in os.listdir(img_folder)]
    for i, dir in enumerate(dirNames):
        if os.path.isdir(dir) == False:
            continue
        print("dir name : " + dir)
        frames = []
        fileNames = [file for file in os.listdir(dir) if file.endswith('jpg')]
        fileNames.sort()
        cnt = 0
        for j, f in enumerate(fileNames):
            full_filename = os.path.join(dir, f)
            if full_filename.endswith('.jpg'):
                if j % frame_rate == 0:
                    frame = io.imread(full_filename)
                    frames.append(frame)
        frames = np.array(frames)
        frames_pad = np.zeros((1, max_frame_num, frames.shape[1], frames.shape[2], frames.shape[3]))
        frames = np.reshape(frames, (1, frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        for j in range(frames.shape[0]):
            frames_pad[0, j] = frames[0, j]
        if i == 0:
            videos = frames_pad
        else:
            videos = np.concatenate((videos, frames_pad), axis=0)
        if debug == 1 and video_num >= debug_num - 1:
            break
        video_num += 1

    videos = np.moveaxis(videos, -1, 2)
    print("all videos shape")
    print(videos.shape)
    all_labels = read_labels_p3(img_folder, label_folder, debug, frame_rate)
    print("all labels shape")
    print(all_labels.shape)
    """
    if load == 0:
        try:
            os.remove(output_filename)
        except OSError:
            pass
        f = h5py.File(output_filename, "w")
        f.create_dataset("frames", data = frames)
    elif load == 1:
        print("read frames")
        f = h5py.File(output_filename, "r")
        frames = f['frames'][:]
    """
    if debug == 1:
        data = [(videos[i], all_labels[i]) for i in range(debug_num)]
    else:
        data = [(videos[i], all_labels[i]) for i in range(videos.shape[0])]
    return data

def read_labels_p3(img_folder, label_folder, debug, frame_rate):
    #label_list = [file for file in os.listdir(label_folder) if file.endswith('txt')]
    label_list = [file for file in os.listdir(img_folder)]
    all_labels = np.zeros((1, max_frame_num, n_class))
    for i, f in enumerate(label_list):
        full_filename = os.path.join(label_folder, f) + '.txt'
        if os.path.isfile(full_filename) == False:
            continue
        print(full_filename)
        file = open(full_filename, "r")
        cnt = 0
        labels = np.zeros((max_frame_num, n_class))
        for j, line in enumerate(file):
            if line == '\n':
                continue
            if j % frame_rate != 0:
                continue
            #label.append(int(line[:-1]))
            label = np.zeros((1, n_class))
            label[0][int(line[:-1])] = 1
            if j == 0:
                labels = label
            else:
                labels = np.concatenate((labels, label), axis=0)

        labels_pad = np.zeros((1, max_frame_num, n_class))
        labels = np.reshape(labels, (1, labels.shape[0], labels.shape[1]))

        for j in range(labels.shape[1]):
            labels_pad[0, j] = labels[0, j]
        if i == 0:
            all_labels = labels_pad
        else:
            all_labels = np.concatenate((all_labels, labels_pad), axis=0)
        if debug == 1 and i >= debug_num - 1:
            break
    print("all labels shape")
    print(all_labels.shape)
    return all_labels

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


def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1, frame_num=4):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''
    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    cnt = 0
    for frameIdx, frame in enumerate(videogen):
        cnt += 1
        frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
        frames.append(frame)

    downsample_factor = int(cnt / frame_num)
    downsample_frame = []
    for i in range(len(frames)):
        if i % downsample_factor == 0:
            downsample_frame.append(frames[i])
    while len(downsample_frame) < frame_num:
        frames.append(frames[len(frames) - 1])
    downsample_frame = downsample_frame[0: frame_num]
    return np.array(downsample_frame[0: frame_num]).astype(np.uint8)


def getVideoList(data_path):
    '''
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    '''
    result = {}

    with open (data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od
