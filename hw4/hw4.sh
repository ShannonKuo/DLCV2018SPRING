#!bin/bash

python3 vae_pytorch.py test $1 $2
python3 gan.py test $1 $2
python3 acgan.py test $1 $2
