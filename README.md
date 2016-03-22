# Train DeepLab for Semantic Image Segmentation

Martin Kersner, <m.kersner@gmail.com>

This repository contains scripts for training [DeepLab for Semantic Image Segmentation](https://bitbucket.org/deeplab/deeplab-public) using strongly annotated data.

```bash
git clone --recursive https://github.com/martinkersner/train-DeepLab.git 
```

In following tutorial we use couple of shell variables in order to reproduce the same results without any obtacles.
* *$DEEPLAB* denotes the main directory where repository is checked out
* *$DATASETS* denotes path to directory where all necessary datasets are stored
* *$LOGNAME* denotes name of log file stored in *$DEEPLAB/exper/voc12/log* directory

## Prerequisites
* [matio](http://sourceforge.net/projects/matio/files/matio/1.5.2/)

### Install DeepLab caffe
You should [follow instructions](http://caffe.berkeleyvision.org/installation.html) for installation.
However, if you have already fulfilled all necessary [dependencies](http://caffe.berkeleyvision.org/installation.html#prerequisites) running following commands from *code/* directory should do the job. 

```bash
cd $DEEPLAB/code
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
make all
make pycaffe
make test # NOT mandatory
make runtest # NOT mandatory
```

### Compile DenseCRF
Go to *$DEEPLAB/code/densecrf* directory, modify *Makefile* if necessary and run *make* command.
Or you can run following commands in sequential order.

```bash
cd $DEEPLAB/code/densecrf
# Adjust Makefile if necessary
make
```

## Dataset
All necessary data for training are listed in [$DEEPLAB/exper/voc12/list/original](https://github.com/martinkersner/train-DeepLab/tree/master/exper/voc12/list/original).
Training scripts are prepared to employ either [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) or [augmented PASCAL VOC dataset](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html) which contains more images.

```bash
# augmented PASCAL VOC
cd $DATASETS
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
tar -zxvf benchmark.tgz
mv benchmark_RELEASE VOC_aug

# original PASCAL VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012 VOC2012_orig && rm -r VOCdevkit
```

### Data conversions
Unfortunately, ground truth labels within augmented PASCAL VOC dataset are distributed as Matlab data files, therefore we will have to convert them before we can start training itself.

```bash
cd $DATASETS/VOC_aug/dataset
mkdir cls_png
cd $DEEPLAB
./mat2png.py $DATASETS/VOC_aug/dataset/cls $DATASETS/VOC_aug/dataset/cls_png
```

Caffe softmax loss function can accept only one-dimensional ground truth labels. However, those labels in original PASCAL VOC 2012 dataset are defined as RGB images. Thus, we have to reduce their dimensionality.

```bash
cd $DATASETS/VOC2012_orig
mkdir SegmentationClass_1D

cd $DEEPLAB
./convert_labels $DATASETS/VOC2012_orig/SegmentationClass/ \
  $DATASETS/VOC2012_orig/ImageSets/Segmentation/trainval.txt \
  $DATASETS/VOC2012_orig/SegmentationClass_1D/
```

At last, part of code which computes DenseCRF is able to work only with PPM image files, hence we have to perform another conversion.

```bash
cd $DEEPLAB

# augmented PASCAL VOC
mkdir $DATASETS/VOC_aug/dataset/img_ppm
./jpg2ppm.sh $DATASETS/VOC_aug/dataset/img $DATASETS/VOC_aug/dataset/img_ppm

# original PASCAL VOC 2012
mkdir $DATASETS/VOC2012_orig/PPMImages
./jpg2ppm.sh $DATASETS/VOC2012_orig/JPEGImages $DATASETS/VOC2012_orig/PPMImages
```

### Connect $DATASETS into $DEEPLAB
Then we create symbolic links to training images and ground truth labels.

```bash
mkdir -p $DEEPLAB/exper/voc12/data
cd $DEEPLAB/exper/voc12/data

# augmented PASCAL VOC
ln -s $DATASETS/VOC_aug/dataset/img images_aug
ln -s $DATASETS/VOC_aug/dataset/cls_png labels_aug
ln -s $DATASETS/VOC_aug/dataset/img_ppm images_aug_ppm

# original PASCAL VOC 2012
ln -s $DATASETS/VOC2012_orig/JPEGImages images_orig
ln -s $DATASETS/VOC2012_orig/SegmentationClass_1D labels_orig
ln -s $DATASETS/VOC2012_orig/PPMImages images_orig_ppm
```

## Training
Before the first training we have to download several files. Using the command below we download initialization model, definition its network and solver. It will also setup symbolic links in directories where those files are later expected during training.

```bash
./get_DeepLab_LargeFOV_voc12_data.sh
```

In order to easily switch between datasets we will modify image lists appropriately.

```bash
./prepare_voc12_data_lists.sh
```

Finally, we can start training.

```bash
./run_pascal.sh
```

Training script generates information which are printed to terminal and also stored in *$DEEPLAB/exper/voc12/log* directory.
For every printed iteration there are displayed loss and three different model evalutation metrics for currently employed batch.
They denote pixel accuracy, average recall and average Jacard index, respectively.
Even though those values are retrievd from training data, they possess important information about training and using the script below we can plot them as a graph.
The script generates two graphs *evaluation.png* and *loss.png*.

```bash
cd $DEEPLAB
./loss_from_log.py exper/voc12/log/DeepLab-LargeFOV/`ls -t exper/voc12/log/DeepLab-LargeFOV/ | head -n 1` # for the newest log
#./loss_from_log.py exper/voc12/log/DeepLab-LargeFOV/$LOGNAME # specified log 
```

### Note 
Init models are modified VGG-16 networks with changed kernel size from 7x7 to 4x4 or 3x3.
There are two models that can be employed for initialization: vgg16_128, vgg16_20M.

The first fully connected layer of [vgg16_128](http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_128.caffemodel) has kernel size 4x4 and 4096 filters. It can be used for DeepLab basic model.
In [vgg16_20M](http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel), the first fully connected layer has kernel size 3x3 and 1024 filters. It can be used for DeepLab-LargeFOV.

Currently training is focused on DeepLab-LargeFOV.

## FAQ
At [http://ccvl.stat.ucla.edu/deeplab_faq/](http://ccvl.stat.ucla.edu/deeplab_faq/) you can find frequently asked questions about DeepLab for Semantic Image Segmentation.
