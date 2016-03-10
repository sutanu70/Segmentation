#!/usr/bin/env sh
# Martin Kersner, m.kersner@gmail.com
# 2016/03/10

# DeepLab-LargeFOV
NET_ID=DeepLab-LargeFOV 
MODEL_PATH=exper/voc12/model/${NET_ID}
CONFIG_PATH=exper/voc12/config/${NET_ID}

mkdir -p ${MODEL_PATH}
mkdir -p ${CONFIG_PATH}

cd ${MODEL_PATH}

wget http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/solver.prototxt
wget http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/solver2.prototxt
wget http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/test.prototxt
wget http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/train.prototxt
wget http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/train2_iter_8000.caffemodel

ln -s train2_iter_8000.caffemodel init.caffemodel

cd ../../config/${NET_ID}
ln -s ../../model/${NET_ID}/solver.prototxt solver.prototxt
ln -s ../../model/${NET_ID}/train.prototxt train.prototxt

cd ../../../../
