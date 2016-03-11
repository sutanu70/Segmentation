#!/usr/bin/env sh
# Martin Kersner, m.kersner@gmail.com
# 2016/03/10

# DeepLab-LargeFOV
NET_ID=DeepLab-LargeFOV 
INIT_PATH=init
MODEL_PATH=exper/voc12/model/${NET_ID}
CONFIG_PATH=exper/voc12/config/${NET_ID}

mkdir -p ${NET_ID}
mkdir -p ${INIT_PATH}
mkdir -p ${MODEL_PATH}
mkdir -p ${CONFIG_PATH}

cd ${NET_ID}

wget -nc http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/solver.prototxt
wget -nc http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/solver2.prototxt
wget -nc http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/test.prototxt
wget -nc http://ccvl.stat.ucla.edu/ccvl/DeepLab-LargeFOV/train.prototxt

cd ../${INIT_PATH}
wget -nc http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel

cd ../${MODEL_PATH}
ln -s ../../../../${INIT_PATH}/vgg16_20M.caffemodel init.caffemodel

cd ../../config/${NET_ID}
ln -s ../../../../${NET_ID}/solver.prototxt solver.prototxt
ln -s ../../../../${NET_ID}/train.prototxt train.prototxt

cd ../../../../
