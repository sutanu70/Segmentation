#!/usr/bin/env sh
# Martin Kersner, m.kersner@gmail.com
# 2016/04/07 

# TODO store init model and DeepLab-LargeFOV_Semi_Bbox_Fixed files in one directory

NET_ID=DeepLab-LargeFOV-Semi-Bbox-Fixed
INIT_PATH=init
MODEL_PATH=exper/voc12/model/${NET_ID}
CONFIG_PATH=exper/voc12/config/${NET_ID}

mkdir -p ${INIT_PATH}
mkdir -p ${MODEL_PATH}
mkdir -p ${CONFIG_PATH}

cd ${INIT_PATH}
wget -nc http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel

cd ../${MODEL_PATH}
ln -s ../../../../${INIT_PATH}/vgg16_20M.caffemodel init.caffemodel

cd ../../config/${NET_ID}
ln -s ../../../../${NET_ID}/solver.prototxt solver.prototxt
ln -s ../../../../${NET_ID}/solver2.prototxt solver2.prototxt
ln -s ../../../../${NET_ID}/train.prototxt train.prototxt
ln -s ../../../../${NET_ID}/test.prototxt test.prototxt

cd ../../../../
