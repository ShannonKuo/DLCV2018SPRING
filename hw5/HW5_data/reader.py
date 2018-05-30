import numpy as np
import skvideo.io
import skimage.transform
import csv
import collections
import os

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
