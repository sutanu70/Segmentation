#!/usr/bin/env sh
# Martin Kersner, m.kersner@gmail.com
# 2016/03/10

LIST_DIR=exper/voc12/list

rm ${LIST_DIR}/*.txt
cp ${LIST_DIR}/original/*.txt ${LIST_DIR}

# original VOC PASCAL 2012
for pfile in "val.txt" "trainval.txt" "train.txt" "test.txt"; do
  FILE_PATH=$LIST_DIR/$pfile
  sed -i 's/JPEGImages/images_orig/g' $FILE_PATH
  sed -i 's/SegmentationClassAug/labels_orig/g' $FILE_PATH
done

# augmented VOC PASCAL
for pfile in "trainval_aug.txt" "train_aug.txt"; do
  FILE_PATH=$LIST_DIR/$pfile
  sed -i 's/JPEGImages/images_aug/g' $FILE_PATH
  sed -i 's/SegmentationClassAug/labels_aug/g' $FILE_PATH
done
