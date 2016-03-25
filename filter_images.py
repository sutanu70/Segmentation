#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/24 

# TODO check if directories and file exist

from __future__ import print_function
import os
import sys
from skimage.io import imread, imsave
import numpy as np
from utils import get_id_classes, convert_from_color_segmentation, create_lut

def main():
  ## 
  ext = '.png'
  class_names = ['bird', 'bottle', 'chair']
  ## 

  input_path, output_path, list_file, subset_data_file = process_arguments(sys.argv)

  clear_subset_list_logs(subset_data_file)
  class_ids = get_id_classes(class_names)
  lut = create_lut(class_ids)

  with open(list_file, 'rb') as f:
    for img_name in f:
      img_name = img_name.strip()
      img = contain_class(os.path.join(input_path, img_name)+ext, class_ids, lut)

      if img != None:
        log_image(img_name, subset_data_file)
        imsave(os.path.join(output_path, img_name)+ext, img)

def clear_subset_list_logs(file_name):
  if os.path.isfile(file_name):
    os.remove(file_name)

def log_image(img_name, list_file):
  with open(list_file, 'ab') as f:
    print(img_name, file=f)

def contain_class(img_name, class_ids, lut):
  img = imread(img_name)

  # If label is three-dimensional image we have to convert it to
  # corresponding labels (0 - 20). Currently anticipated labels are from
  # VOC pascal datasets.
  if (len(img.shape) > 2):
    img = convert_from_color_segmentation(img)

  img_labels = np.unique(img)

  if len(set(img_labels).intersection(class_ids)) >= 1:
    return lut[img] 
  else:
    return None

def process_arguments(argv):
  if len(argv) != 5:
    help()

  input_path       = argv[1]
  output_path      = argv[2]
  list_file        = argv[3]
  subset_list_file = argv[4]

  return input_path, output_path, list_file, subset_list_file

def help():
  print('Usage: python filter_images.py INPUT_PATH OUTPUT_PATH LIST_FILE SUBSET_LIST_FILE\n'
        'INPUT_PATH points to directory with segmentation ground truth labels.\n'
        'OUTPUT_PATH point to directory where reindexed ground truth labels are going to be stored.\n'
        'LIST_FILE denotes text file containing names of images in INPUT_PATH.\n'
        'SUBSET_LIST_FILE denotes text file with remaining images that contain specified labels.\n'
        'Names do not include extension of images.'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
