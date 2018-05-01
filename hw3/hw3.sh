#!bin/bash
wget https://www.dropbox.com/s/iqrg2smua5029xm/baseline.h5?dl=1
python3 train.py baseline.h5?dl=1 test base $1 $2
