#!bin/bash
wget https://www.dropbox.com/s/li5zwe82d5vmbpg/best.h5?dl=1 
python3 train.py best.h5?dl=1 test best $1 $2
