# Train DeepLab for Semantic Image Segmentation

Martin Kersner, <m.kersner@gmail.com>

This repository contains scripts for training [DeepLab for Semantic Image Segmentation](https://bitbucket.org/deeplab/deeplab-public) using [strongly](https://github.com/martinkersner/train-DeepLab#strong-annotations) and [weakly annotated data](https://github.com/martinkersner/train-DeepLab#weak-annotations).
[Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](http://arxiv.org/abs/1412.7062) and [Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation](http://arxiv.org/abs/1502.02734) papers describe training procedure using strongly and weakly annotated data, respectively.

```bash
git clone --recursive https://github.com/martinkersner/train-DeepLab.git 
```

In following tutorial we use couple of shell variables in order to reproduce the same results without any obtacles.
* *$DEEPLAB* denotes the main directory where repository is checked out
* *$DATASETS* denotes path to directory where all necessary datasets are stored
* *$LOGNAME* denotes name of log file stored in *$DEEPLAB/exper/voc12/log* directory
* *$DOWNLOADS* denotes directory where downloaded files are stored

## Prerequisites
* [matio](http://sourceforge.net/projects/matio/files/matio/1.5.2/)

## Install DeepLab caffe
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

## Compile DenseCRF
Go to *$DEEPLAB/code/densecrf* directory, modify *Makefile* if necessary and run *make* command.
Or you can run following commands in sequential order.

```bash
cd $DEEPLAB/code/densecrf
# Adjust Makefile if necessary
make
```

## Strong annotations
In this part of tutorial we train DCNN for semantic image segmentation using PASCAL VOC dataset with all 21 classes and also with limited number of them.
As a training data we use only strong annotations (pixel level labels).

### Dataset
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

#### Data conversions
Unfortunately, ground truth labels within augmented PASCAL VOC dataset are distributed as Matlab data files, therefore we will have to convert them before we can start training itself.

```bash
cd $DATASETS/VOC_aug/dataset
mkdir cls_png
cd $DEEPLAB
./mat2png.py $DATASETS/VOC_aug/dataset/cls $DATASETS/VOC_aug/dataset/cls_png
```

Caffe softmax loss function can accept only one-channel ground truth labels. However, those labels in original PASCAL VOC 2012 dataset are defined as RGB images. Thus, we have to reduce their dimensionality.

```bash
cd $DATASETS/VOC2012_orig
mkdir SegmentationClass_1D

cd $DEEPLAB
./convert_labels.py $DATASETS/VOC2012_orig/SegmentationClass/ \
  $DATASETS/VOC2012_orig/ImageSets/Segmentation/trainval.txt \
  $DATASETS/VOC2012_orig/SegmentationClass_1D/
```

At last, part of code which computes DenseCRF is able to work only with PPM image files, hence we have to perform another conversion.
This step is necessary only if we want to use DenseCRF separately and as one of Caffe layers.

```bash
cd $DEEPLAB

# augmented PASCAL VOC
mkdir $DATASETS/VOC_aug/dataset/img_ppm
./jpg2ppm.sh $DATASETS/VOC_aug/dataset/img $DATASETS/VOC_aug/dataset/img_ppm

# original PASCAL VOC 2012
mkdir $DATASETS/VOC2012_orig/PPMImages
./jpg2ppm.sh $DATASETS/VOC2012_orig/JPEGImages $DATASETS/VOC2012_orig/PPMImages
```

#### Connect $DATASETS into $DEEPLAB
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

### Download necessary files for training 
Before the first training we have to download several files. Using the command below we download initialization model, definition its network and solver. It will also setup symbolic links in directories where those files are later expected during training.

```bash
./get_DeepLab_LargeFOV_voc12_data.sh
```

In order to easily switch between datasets we will modify image lists appropriately.

```bash
./prepare_voc12_data_lists.sh
```

### Training with all classes
*run_pascal_strong.sh* can go through 4 different phases (twice training, twice testing), but I wouldn't recommend to run testing phases using this script.
Actually, they are currently disabled.
At [lines 27 through 30](https://github.com/martinkersner/train-DeepLab/blob/master/run_pascal_strong.sh#L27-L30), any of phase can be enabled (value 1) or disabled (value 0).

Finally, we can start training.

```bash
./run_pascal_strong.sh
```

#### Plotting training information
Training script generates information which are printed to terminal and also stored in *$DEEPLAB/exper/voc12/log/DeepLab-LargeFOV/* directory.
For every printed iteration there are displayed loss and three different model evalutation metrics for currently employed batch.
They denote pixel accuracy, average recall and average Jacard index, respectively.
Even though those values are retrievd from training data, they possess important information about training and using the script below we can plot them as a graph.
The script generates two graphs *evaluation.png* and *loss.png*.

```bash
cd $DEEPLAB
./loss_from_log.py exper/voc12/log/DeepLab-LargeFOV/`ls -t exper/voc12/log/DeepLab-LargeFOV/ | head -n 1` # for the newest log
#./loss_from_log.py exper/voc12/log/DeepLab-LargeFOV/$LOGNAME # specified log 
```

### Training with only 3 classes
If we want to train with limited number of classes we have to modify ground truth labels and also list of images that can be exploited for training.
In *filter_images.py* at [line 17](https://github.com/martinkersner/train-DeepLab/blob/master/filter_images.py#L17) are specified classes that we are interested in (defaultly bird, bottle and chair).

```bash
# augmented PASCAL VOC 
mkdir -p $DATASETS/VOC_aug/dataset/cls_sub_png
cd $DEEPLAB/exper/voc12/data/
ln -s $DATASETS/VOC_aug/dataset/cls_sub_png labels_sub_aug
find exper/voc12/data/labels_aug/ -printf '%f\n' | sed 's/\.png//'  | tail -n +2 > all_aug_data.txt
python filter_images.py $DATASETS/VOC_aug/dataset/cls_png/ $DATASETS/VOC_aug/dataset/cls_sub_png/ all_aug_data.txt sub_aug_data.txt

# original PASCAL VOC 2012 
mkdir -p $DATASETS/VOC2012_orig/SegmentationClass_sub_1D
cd $DEEPLAB/exper/voc12/data/
ln -s $DATASETS/VOC2012_orig/SegmentationClass_sub_1D labels_sub_orig
find exper/voc12/data/labels_orig/ -printf '%f\n' | sed 's/\.png//'  | tail -n +2 > all_orig_data.txt
python filter_images.py $DATASETS/VOC2012_orig/SegmentationClass_1D/ $DATASETS/VOC2012_orig/SegmentationClass_sub_1D/ all_orig_data.txt sub_orig_data.txt

./filter_lists.sh
```

The number of classes that we plan to use is set at [lines 13 and 14](https://github.com/martinkersner/train-DeepLab/blob/master/run_pascal_strong.sh#L13-L14) in *run_pascal_strong.sh*.
This number should be always higher by 1 than number of specified classes in *filter_images.py* script, because we also consider background as one of classes.

After, we can proceed to training.

```bash
./run_pascal_strong.sh
```

We can also use the same script for [plotting training information](https://github.com/martinkersner/train-DeepLab#plotting-training-information).

#### Evaluation
|                       | phase 1 (24,000 iter., no CRF) | phase 2 (12,000 iter., no CRF) |
|-----------------------|--------------------------------|--------------------------------|
| pixel accuracy        |                         0.8315 |                         0.8523 |
| mean accuracy         |                         0.6807 |                         0.6987 |
| mean IU               |                         0.6725 |                         0.6937 |
| frequency weighted IU |                         0.8182 |                         0.8439 |

#### Visual results
Employed model was trained without CRF in phase 1 (24,000 iterations) and then in phase 2 (12,000 iterations), but results here exploited DENSE_CRF layer.
Displayed images (bird: 2010_004994, bottle: 2007_000346, chair: 2008_000673) are part of validation dataset stored in *$DEEPLAB/exper/voc12/list_subset/val.txt*.
Colors of segments differ from original ground truth labels because employed model was trained only for 3 classes + background.

<p align="center">
<img src="http://i.imgur.com/n5PzvZU.png?1" />
<img src="http://i.imgur.com/DCeUXes.png?1" />
<img src="http://i.imgur.com/68KOfn4.png?1" />
</p>

## Weak annotations
In a case we don't possess enough training data, weakly annotated ground truth labels can be exploited using DeepLab.

### Dataset
At first you should download *SegmentationClassBboxRect_Visualization.zip* and *SegmentationClassBboxSeg_Visualization.zip* from link [https://ucla.app.box.com/s/laif889j7pk6dj04b0ou1apm2sgub9ga](https://ucla.app.box.com/s/laif889j7pk6dj04b0ou1apm2sgub9ga) and run commands below to prepare data for use.

```bash
cd $DOWNLOADS
mv SegmentationClassBboxRect_Visualization.zip $DATASETS/VOC_aug/dataset/
mv SegmentationClassBboxSeg_Visualization.zip $DATASETS/VOC_aug/dataset/

cd $DATASETS/VOC_aug/dataset
unzip SegmentationClassBboxRect_Visualization.zip
unzip SegmentationClassBboxSeg_Visualization.zip

mv SegmentationClassBboxAug_Visualization/ SegClassBboxAug_RGB
mv SegmentationClassBboxErode20CRFAug_Visualization/ SegClassBboxErode20CRFAug_RGB
```

Downloaded weak annotations were created using Matlab and because of that labels are sometimes stored with one channel and other times with three channels.
Similarly to [strong annotations](https://github.com/martinkersner/train-DeepLab#data-conversions) we have to convert all labels to the same one channel format.
In order to cope with it I recommend you to use Matlab script *convert_weak_labels.m* (if anybody knows how to perform the same conversion using python I would be really interested) which is stored in $DEEPLAB directory.
Before running script you have to specify path to datasets on line 3.

After script successfully finished we have to create symbolic links to be able to reach data during training.
```bash
cd $DEEPLAB/exper/voc12/data
ln -s $DATASETS/VOC_aug/dataset/SegClassBboxAug_1D/ labels_bbox
ln -s $DATASETS/VOC_aug/dataset/SegClassBboxErode20CRFAug_1D/ labels_bboxcrf
```

#### Create subsets
Training DeepLab using weak labels enables to employ datasets of different sizes.
Following snippet creates those subsets of strong dataset and also necessary training lists with weak labels.

```bash
cd $DEEPLAB
./create_weak_lists.sh

cd $DEEPLAB/exper/voc12/list
head -n 200  train.txt > train200.txt
head -n 500  train.txt > train500.txt
head -n 750  train.txt > train750.txt
head -n 1000 train.txt > train1000.txt

cp train_bboxcrf.txt trainval_bboxcrf.txt
cp train_bbox.txt trainval_bbox.txt
```

### Training
Training using weak annotations is similar to exploiting [strong annotations](https://github.com/martinkersner/train-DeepLab#training-with-all-classes).
The only difference is the name of script which should be run.

```bash
./run_pascal_weak.sh
```

[Plotting](https://github.com/martinkersner/train-DeepLab#plotting-training-information) is also same as for strong annotations.

#### Evaluation
5000 weak annotations and 200 strong annotations

|                       | phase 1 (6,000 iter., no CRF) | phase 2 (8,000 iter., no CRF) |
|-----------------------|--------------------------------|--------------------------------|
| pixel accuracy        |                         0.8688 |                         0.8671 |
| mean accuracy         |                         0.7415 |                          0.750 |
| mean IU               |                         0.6324 |                         0.6343 |
| frequency weighted IU |                         0.7962 |                         0.7951 |

## Note 
Init models are modified VGG-16 networks with changed kernel size from 7x7 to 4x4 or 3x3.
There are two models that can be employed for initialization: vgg16_128, vgg16_20M.

The first fully connected layer of [vgg16_128](http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_128.caffemodel) has kernel size 4x4 and 4096 filters. It can be used for DeepLab basic model.
In [vgg16_20M](http://ccvl.stat.ucla.edu/ccvl/init_models/vgg16_20M.caffemodel), the first fully connected layer has kernel size 3x3 and 1024 filters. It can be used for DeepLab-LargeFOV.

Currently training is focused on DeepLab-LargeFOV.

## FAQ
At [http://ccvl.stat.ucla.edu/deeplab_faq/](http://ccvl.stat.ucla.edu/deeplab_faq/) you can find frequently asked questions about DeepLab for semantic image segmentation.
