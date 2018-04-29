#!bin/bash

python3 train.py 5 0 best_epoch5_relu_adadelta_lr2.h5 best_epoch5_relu_adadelta_lr2.h5 Adadelta output_best_epoch5_relu_adadelta_lr2.txt relu 2 train best
python3 train.py 5 0 best_epoch5_relu_adadelta_lr0.5.h5 best_epoch5_relu_adadelta_lr0.5.h5 Adadelta output_best_epoch5_relu_adadelta_lr0.5.txt relu 0.5 train best

python3 train.py 5 0 best_epoch5_softmax_adadelta_lr2.h5 best_epoch5_softmax_adadelta_lr2.h5 Adadelta output_best_epoch5_softmax_adadelta_lr2.txt softmax 2 train best
python3 train.py 5 0 best_epoch5_softmax_adadelta_lr0.5.h5 best_epoch5_softmax_adadelta_lr0.5.h5 Adadelta output_best_epoch5_softmax_adadelta_lr0.5.txt softmax 0.5 train best


#baseline train
#python3 train.py 18 0 my_model_epoch20_softmax_lr0.5.h5 my_model_epoch20_lr0.5_softmax.h5 Adadelta output_mean_softmax_lr0.5.txt softmax 0.5 train base

#test
#python3 train.py 18 0 baseline_epoch20_softmax_adadelta_lr0.5.h5 test.h5 Adadelta output.txt softmax 0.5 test base
