#!/usr/bin/env sh
# Martin Kersner, m.kersner@gmail.com
# 2016/03/10

LIST_DIR=exper/voc12/list

rm ${LIST_DIR}/*.txt
cp ${LIST_DIR}/original/*.txt ${LIST_DIR}/

for FILE in ${LIST_DIR}/*.txt; do
 sed -i 's/jpg/png/g' $FILE
 sed -i 's/JPEGImages/images/g' $FILE
 sed -i 's/SegmentationClassAug/labels/g' $FILE
done
