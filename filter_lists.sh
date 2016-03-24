#!/usr/bin/env bash
# Martin Kersner, m.kersner@gmail.com
# 2016/03/24

TXT_EXT=.txt
BASE=exper/voc12
LIST_ALL_DIR=$BASE/list/
LIST_SUB_DIR=$BASE/list_subset/

mkdir -p $LIST_SUB_DIR

SUB_SEGM_LIST=subset_data.txt

for file in test_id  test  train_aug  train  trainval_aug  trainval  val_id  val; do
  LIST_ALL_FILE="$LIST_ALL_DIR""$file""$TXT_EXT"
  LIST_SUB_FILE="$LIST_SUB_DIR""$file""$TXT_EXT"

  grep -f $SUB_SEGM_LIST $LIST_ALL_FILE > $LIST_SUB_FILE

  # for training using limited number of classes only augmented dataset is employed
  sed -i 's/labels_aug/labels_sub_aug/g' $LIST_SUB_FILE
done
