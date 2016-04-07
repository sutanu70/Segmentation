#!/usr/bin/env sh

# Martin Kersner, m.kersner@gmail.com
# 2016/04/07 

EXP=exper/voc12
DATA=$EXP"/data"
LIST=$EXP"/list/"
AUG_IMG=images_aug/
BBOX_LABEL=labels_bbox/
BBOXCRF_LABEL=labels_bboxcrf/
BBOX_DIR=$DATA"/"$BBOX_LABEL
AUG_DIR=$DATA"/"$AUG_IMG
BBOX_LIST_PATH=$LIST"train_bbox.txt"
BBOXCRF_LIST_PATH=$LIST"train_bboxcrf.txt"

for label_name in `find $BBOX_DIR -printf '%f\n' | tail -n +2`; do

  label_name_no_ext=`echo $label_name | sed s@.png@@`
  NUM=`ls $AUG_DIR$label_name_no_ext* 2> /dev/null | wc -l`

  if [ $NUM -ne 0 ]; then
    echo "/"$AUG_IMG$label_name_no_ext".jpg /"$BBOX_LABEL$label_name >> $BBOX_LIST_PATH
    echo "/"$AUG_IMG$label_name_no_ext".jpg /"$BBOXCRF_LABEL$label_name >> $BBOXCRF_LIST_PATH
  fi

done
