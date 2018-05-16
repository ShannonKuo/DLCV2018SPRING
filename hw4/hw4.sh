#!bin/bash

python3 vae_pytorch.py train $1 $2
python3 gan.py train $1 $2
python3 acgan.py train $1 $2
