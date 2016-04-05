#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/18

from __future__ import print_function
caffe_root = 'code/'
import sys
sys.path.insert(0, caffe_root + 'python')
import os
import caffe
import numpy as np
from skimage.io import imread
from py_img_seg_eval.eval_segm import *
from utils import get_id_classes, create_lut
from segmenter import Segmenter
from ProgressBar import * 

def main():
  iteration_num, phase, subset_dataset = process_arguments(sys.argv)
  model_name = 'DeepLab-LargeFOV'
  base_dir = 'exper/voc12'
  gpu_id = 2

  if phase == 1:
    model_path = os.path.join(base_dir, 'model', model_name, 'train_iter_{}.caffemodel')
  elif phase == 2:
    model_path = os.path.join(base_dir, 'model', model_name, 'train2_iter_{}.caffemodel')

  if subset_dataset:
    net_path = os.path.join(model_name, 'deploy4.prototxt')
    class_names = ['bird', 'bottle', 'chair'] # CHANGE
    class_ids = get_id_classes(class_names)
    file_names = load_test_data(os.path.join(base_dir, 'list_subset/val_id.txt'))
    images_path = os.path.join(base_dir, 'data/images_orig')
    labels_path = os.path.join(base_dir, 'data/labels_sub_orig')
  else:
    net_path = os.path.join(model_name, 'deploy21.prototxt')
    class_ids = range(1,21)
    file_names = load_test_data(os.path.join(base_dir, 'list/val_id.txt'))
    images_path = os.path.join(base_dir, 'data/images_orig')
    labels_path = os.path.join(base_dir, 'data/labels_orig')

  lut = create_lut(class_ids)
  images, labels = create_full_paths(file_names, images_path, labels_path)
  test_net(net_path, model_path.format(iteration_num), images, labels, lut, gpu_id)

def load_test_data(file_name='test.txt'):
  file_names = []
  with open(file_name, 'rb') as f:
      for fn in f:
          file_names.append(fn.strip())

  return file_names

def create_full_paths(file_names, image_dir, label_dir, image_ext='.jpg', label_ext='.png'):
  image_paths = []
  label_paths = []

  for file_name in file_names:
    image_paths.append(os.path.join(image_dir, file_name+image_ext))
    label_paths.append(os.path.join(label_dir, file_name+label_ext))

  return image_paths, label_paths

def preprocess_image(file_name, mean_vec, height=500, width=500):
  image = imread(file_name).astype(np.float32)
  im = image[:,:,::-1]
  im = im - mean_vec
  cur_h, cur_w, cur_c = im.shape
  pad_h = height - cur_h
  pad_w = width  - cur_w
  im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

  return im, cur_h, cur_w

def test_net(net_path, model_path, images, labels, lut, gpu_id):
  net = Segmenter(net_path, model_path, gpu_id)

  mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
  reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

  pa_list    = []
  ma_list    = []
  m_IU_list  = []
  fw_IU_list = []

  pb = ProgressBar(len(images))

  for img_path, label_path in zip(images, labels):
    im, cur_h, cur_w = preprocess_image(img_path, reshaped_mean_vec)
    label = imread(label_path)
    label = lut[label]

    segmentation = net.predict([im])
    pred = segmentation[0:cur_h, 0:cur_w]
   
    pa = pixel_accuracy(pred, label)
    ma = mean_accuracy(pred, label)
    m_IU = mean_IU(pred, label)
    fw_IU = frequency_weighted_IU(pred, label)

    pa_list.append(pa)
    ma_list.append(ma)
    m_IU_list.append(m_IU)
    fw_IU_list.append(fw_IU)

    pb.print_progress()

  print("pixel_accuracy: " + str(np.mean(pa_list)))
  print("mean_accuracy: " + str(np.mean(ma_list)))
  print("mean_IU: " + str(np.mean(m_IU_list)))
  print("frequency_weighted: " + str(np.mean(fw_IU_list)))

def process_arguments(argv):
  if len(argv) != 4 or (int(argv[2]) != 1 and int(argv[2]) != 2) or (int(argv[3]) != 0 and int(argv[3]) != 1):
    help()

  iteration_num  = argv[1]
  phase          = int(argv[2])
  subset_dataset = bool(int(argv[3]))

  return iteration_num, phase, subset_dataset

def help():
  print('Usage: python test_model.py ITERATION_NUM PHASE SUBSET_DATASET\n'
        'ITERATION_NUM denotes iteration number of model which will be tested.'
        'PHASE denotes training phase (either 1 or 2) that should be tested.\n'
        'SUBSET_DATASET determines whether subset of whole dataset should be used (value 1) or whole dataset will be exploited (value 0).'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
