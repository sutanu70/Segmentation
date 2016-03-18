#!/usr/bin/env python

# Martin Kersner, m.kersner@gmail.com
# 2015/12/09

# pb = ProgressBar(count)
# pb.print_progress()

import sys
import numpy as np

class ProgressBar:
  def __init__(self, count):
    self.count = count
    self.progress = 0
    self.step = 100.0/count

    self.decimals = 2

  def print_progress(self):
    self.progress += self.step
    print str(np.round(self.progress, decimals=self.decimals)) + " %",
    sys.stdout.flush()
    print "\r",
