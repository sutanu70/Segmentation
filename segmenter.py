#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/11

# Segmenter is an image segmentation specialization of Net.
# Inspired by https://github.com/torrvision/crfasrnn/blob/master/caffe-crfrnn/python/caffe/segmenter.py

import numpy as np

caffe_root = 'code/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

class Segmenter(caffe.Net):
  def __init__(self, prototxt, model, gpu_id=-1):
    caffe.Net.__init__(self, prototxt, model)
    self.set_phase_test()

    if gpu_id < 0:
      self.set_mode_cpu()
    else:
      self.set_mode_gpu()
      self.set_device(gpu_id)

  def predict(self, inputs):
    # uses MEMORY_DATA layer for loading images and postprocessing DENSE_CRF layer
    img = inputs[0].transpose((2, 0, 1))
    img = img[np.newaxis, :].astype(np.float32)
    label = np.zeros((1, 1, 1, 1), np.float32)
    data_dim = np.zeros((1, 1, 1, 2), np.float32)
    data_dim[0][0][0][0] = img.shape[2]
    data_dim[0][0][0][1] = img.shape[3]

    img      = np.ascontiguousarray(img, dtype=np.float32)
    label    = np.ascontiguousarray(label, dtype=np.float32)
    data_dim = np.ascontiguousarray(data_dim, dtype=np.float32)

    self.set_input_arrays(img, label, data_dim)
    out = self.forward()

    predictions = out[self.outputs[0]] # the output layer should be called crf_inf
    segm_result = predictions[0].argmax(axis=0).astype(np.uint8)

    return segm_result 
