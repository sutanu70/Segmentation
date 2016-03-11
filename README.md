# Train DeepLab for Semantic Image Segmentation

Martin Kersner, <m.kersner@gmail.com>

This repository contains scripts necessary for training [DeepLab for Semantic Image Segmentation](https://bitbucket.org/deeplab/deeplab-public). 

```bash
git clone --recursive https://github.com/martinkersner/train-DeepLab.git 
```

## Prerequisites
* [matio](http://sourceforge.net/projects/matio/files/matio/1.5.2/)

### Install DeepLab caffe
You should [follow instructions](http://caffe.berkeleyvision.org/installation.html) for installation.
However, if you have already fulfilled all necessary [dependencies](http://caffe.berkeleyvision.org/installation.html#prerequisites) running following commands from *code/* directory should do the job. 

```bash
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
make all
make pycaffe
make test # NOT mandatory
make runtest # NOT mandatory
```

## Dataset
In order to be able to access [all images necessary for training, validating and testing](https://github.com/martinkersner/train-DeepLab/tree/master/exper/voc12/list/original) you have to download [extended PASCAL VOC dataset](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html).

## Training
Before the first training we have to download several files. Using the following command we download initialization model, definition its network and solver. It will also setup symbolic links in directories where those files are later expected while training.
No matter where you save your dataset, you should create symbolic links for both images and labels in *exper/voc12/data* directory.

```bash
./get_DeepLab_LargeFOV_voc12_data.sh
```

Image list have to be modified appropriately.
```bash
./prepare_voc12_data_lists.sh
```

And then training can start.
```bash
./run_pascal.sh
```

### Note 
Init models are modified VGG-16 networks with changed kernel size from 7×7 to 4×4 or 3×3.
There are two models that can be employed for initialization: vgg16_128, vgg16_20M.

The first fully connected layer of [vgg16_128](http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_128.caffemodel) has kernel size 4×4 and 4096 filters. It can be used for DeepLab basic model.
In [vgg16_20M](http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel), the first fully connected layer has kernel size 3×3 and 1024 filters. It can be used for DeepLab-LargeFOV.

Currently training is focused on DeepLab-LargeFOV.

## FAQ
At [http://ccvl.stat.ucla.edu/deeplab_faq/](http://ccvl.stat.ucla.edu/deeplab_faq/) you can find frequently asked questions about DeepLab for Semantic Image Segmentation.
