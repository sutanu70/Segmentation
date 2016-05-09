#!/usr/bin/env python
# Martin Kersner, 2016/03/11 

from __future__ import print_function
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

from utils import strstr

def main():
  output_data, log_files = process_arguments(sys.argv)

  train_iteration = []
  train_loss      = []
  train_accuracy0 = []
  train_accuracy1 = []
  train_accuracy2 = []
  train_accuracy3 = []
  train_accuracy4 = []
  train_accuracy5 = []

  base_train_iter = 0

  for log_file in log_files:
    with open(log_file, 'rb') as f:
      if len(train_iteration) != 0:
        base_train_iter = train_iteration[-1]

      for line in f:
        if strstr(line, 'Iteration') and strstr(line, 'loss'):
          matched = match_loss(line)
          train_loss.append(float(matched.group(1)))

          matched = match_iteration(line)
          train_iteration.append(int(matched.group(1))+base_train_iter)

        # strong labels
        elif strstr(line, 'Train net output #0: accuracy '):
          matched = match_net_accuracy(line)
          train_accuracy0.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #1: accuracy '):
          matched = match_net_accuracy(line)
          train_accuracy1.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #2: accuracy '):
          matched = match_net_accuracy(line)
          train_accuracy2.append(float(matched.group(1)))

        # weak labels
        elif strstr(line, 'Train net output #0: accuracy_bbox'):
          matched = match_net_accuracy_bbox(line)
          train_accuracy0.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #1: accuracy_bbox'):
          matched = match_net_accuracy_bbox(line)
          train_accuracy1.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #2: accuracy_bbox'):
          matched = match_net_accuracy_bbox(line)
          train_accuracy2.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #3: accuracy_strong'):
          matched = match_net_accuracy_strong(line)
          train_accuracy3.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #4: accuracy_strong'):
          matched = match_net_accuracy_strong(line)
          train_accuracy4.append(float(matched.group(1)))

        elif strstr(line, 'Train net output #5: accuracy_strong'):
          matched = match_net_accuracy_strong(line)
          train_accuracy5.append(float(matched.group(1)))

  
  if output_data == 'loss':
    for x in train_loss:
      print(x)

  if output_data == 'acc1':
    for x,y,z in zip(train_accuracy0, train_accuracy1, train_accuracy2):
      print(x, y, z)

  if output_data == 'acc2':
    for x,y,z in zip(train_accuracy3, train_accuracy4, train_accuracy5):
      print(x, y, z)

  ## loss
  plt.plot(train_iteration, train_loss, 'k', label='Train loss')
  plt.legend()
  plt.ylabel('Loss')
  plt.xlabel('Number of iterations')
  plt.savefig('loss.png')

  ## evaluation
  plt.clf()
  if len(train_accuracy3) != 0:
    plt.plot(range(len(train_accuracy0)), train_accuracy0, 'k', label='accuracy bbox 0')
    plt.plot(range(len(train_accuracy1)), train_accuracy1, 'r', label='accuracy bbox 1')
    plt.plot(range(len(train_accuracy2)), train_accuracy2, 'g', label='accuracy bbox 2')
    plt.plot(range(len(train_accuracy3)), train_accuracy3, 'b', label='accuracy strong 0')
    plt.plot(range(len(train_accuracy4)), train_accuracy4, 'c', label='accuracy strong 1')
    plt.plot(range(len(train_accuracy5)), train_accuracy5, 'm', label='accuracy strong 2')
  else:
    plt.plot(range(len(train_accuracy0)), train_accuracy0, 'k', label='train accuracy 0')
    plt.plot(range(len(train_accuracy1)), train_accuracy1, 'r', label='train accuracy 1')
    plt.plot(range(len(train_accuracy2)), train_accuracy2, 'g', label='train accuracy 2')

  plt.legend(loc=0)
  plt.savefig('evaluation.png')

def match_iteration(line):
  return re.search(r'Iteration (.*),', line)

def match_loss(line):
  return re.search(r'loss = (.*)', line)

def match_net_accuracy(line):
  return re.search(r'accuracy = (.*)', line)

def match_net_accuracy_bbox(line):
  return re.search(r'accuracy_bbox = (.*)', line)

def match_net_accuracy_strong(line):
  return re.search(r'accuracy_strong = (.*)', line)

def process_arguments(argv):
  if len(argv) < 3:
    help()
    
  output_data = None
  log_files = argv[2:]

  if argv[1].lower() == 'loss':
    output_data = 'loss'
  elif argv[1].lower() == 'acc1':
    output_data = 'acc1'
  elif argv[1].lower() == 'acc2':
    output_data = 'acc2'
  else:
    log_files = argv[1:]

  return output_data, log_files

def help():
  print('Usage: python loss_from_log.py OUTPUT_TYPE [LOG_FILE]+\n'
        'OUTPUT_TYPE can be either loss, acc1 or acc 2\n'
        'LOG_FILE is text file containing log produced by caffe.\n'
        'At least one LOG_FILE has to be specified.\n'
        'Files has to be given in correct order (the oldest logs as the first ones).'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
