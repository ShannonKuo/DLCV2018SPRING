#!bin/bash

wget https://www.dropbox.com/s/f4wium8x92932x2/baseline.h5?dl=1
python3 train.py baseline.h5?dl=1 test base $1 $2
