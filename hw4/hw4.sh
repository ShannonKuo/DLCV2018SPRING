#!bin/bash
wget -O vae.pth https://www.dropbox.com/s/yjhdb22dxd2lz92/vae.pth?dl=1
python3 vae_pytorch.py test $1 $2
python3 gan.py test $1 $2
python3 acgan.py test $1 $2
