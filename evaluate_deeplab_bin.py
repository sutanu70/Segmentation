#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/23 

from __future__ import print_function
import os
import sys
import glob
from skimage.io import imread

from py_img_seg_eval.eval_segm import *
from utils import load_binary_segmentation
from ProgressBar import * 

def main():
  list_path, gt_path, result_path = process_arguments(sys.argv)

  gt_ext     = '.png'
  result_ext = '.bin'

  pa_list    = []
  ma_list    = []
  m_IU_list  = []
  fw_IU_list = []

  list_images = load_list(list_path)
  pb = ProgressBar(len(list_images))

  for image_name in list_images:
    gt_fullpath = os.path.join(gt_path, image_name) + gt_ext
    result_fullpath = os.path.join(result_path, image_name) + result_ext

    label = load_binary_segmentation(result_fullpath, dtype='int16')
    pred = imread(gt_fullpath)

    pa_list.append(pixel_accuracy(pred, label))
    ma_list.append(mean_accuracy(pred, label))
    m_IU_list.append(mean_IU(pred, label))
    fw_IU_list.append(frequency_weighted_IU(pred, label))

    pb.print_progress()

  print("pixel_accuracy: "     + str(np.mean(pa_list)))
  print("mean_accuracy: "      + str(np.mean(ma_list)))
  print("mean_IU: "            + str(np.mean(m_IU_list)))
  print("frequency_weighted: " + str(np.mean(fw_IU_list)))

def load_list(list_path):
  list_data = []

  with open(list_path, 'rb') as f:
    for line in f:
      list_data.append(line.strip())

  return list_data

def process_arguments(argv):
  list_path   = None
  gt_path     = None
  result_path = None

  if len(argv) == 4:
    list_path   = argv[1]
    gt_path     = argv[2]
    result_path = argv[3]
  else:
    help()

  if not os.path.exists(list_path):
    help('Given LIST_PATH does not exist!\n')
  if not os.path.exists(gt_path):
    help('Given GT_PATH does not exist!\n')
  if not os.path.exists(result_path):
    help('Given RESULT_PATH does not exist!\n')

  return list_path, gt_path, result_path 

def help(msg=''):
  print(msg +
        'Usage: python evaluate_deeplab_bin.py LIST_PATH GT_PATH RESULT_PATH\n'
        'LIST_PATH denotes path to text file with list of images for evaluating.\n'
        'Gt_PATH denotes path to ground truth labels that will be used for evaluation.\n'
        'RESULT_PATH denotes path to segmentation results that will be evaluated.'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
