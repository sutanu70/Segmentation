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
    """
    Assume that the input is a 500 x 500 image BRG layout with
    correct padding as necessary to make it 500 x 500.
    TODO arbitrary input
    """
      
    input_ = np.zeros((len(inputs), 500, 500, inputs[0].shape[2]), dtype=np.float32)

    for ix, in_ in enumerate(inputs):
      input_[ix] = in_

    # Segment
    caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, in_ in enumerate(input_):
      caffe_in[ix] = in_.transpose((2, 0, 1))

    out = self.forward_all(**{self.inputs[0]: caffe_in})
    predictions = out[self.outputs[0]]

    segm_result = predictions[0].argmax(axis=0).astype(np.uint8)

    return segm_result 
