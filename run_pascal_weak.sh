#!/usr/bin/env sh

# Modified version from https://bitbucket.org/deeplab/deeplab-public using weak annotations
# Martin Kersner, m.kersner@gmail.com
# 2016/04/07 

CAFFE_BIN=code/.build_release/tools/caffe.bin
EXP=exper/voc12
DATA_ROOT=exper/voc12/data
GPU_ID=0

# Specify number of classes
NUM_LABELS=21          # all classes
#NUM_LABELS=4          # 3 classes + 1 background

LIST_SUFFIX=           # all classes
#LIST_SUFFIX=_subset   # only for limited number of classes


# Specify which dataset use for training
#TRAIN_SET_SUFFIX=          # original PASCAL VOC 2012 dataset
#TRAIN_SET_SUFFIX=_aug      # augmented PASCAL VOC dataset
TRAIN_SET_SUFFIX=_bbox      # weak labels consisting of only bounding boxes
#TRAIN_SET_SUFFIX=_bboxcrf  # weak labels consisting of bounding boxes processed by DenseCRF

#TRAIN_SET_STRONG=train
TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train750
#TRAIN_SET_STRONG=train1000

TRAIN_SET_WEAK_LEN=5000

# Specify which model to train
NET_ID=DeepLab-LargeFOV-Semi-Bbox-Fixed

# Run
RUN_TRAIN=1
RUN_TEST=0
RUN_TRAIN2=1
RUN_TEST2=0

# Create directories ###########################################################
CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
LOG_DIR=${EXP}/log/${NET_ID}

for DIR in $CONFIG_DIR $MODEL_DIR $LOG_DIR; do 
  mkdir -p ${DIR}
done

export GLOG_log_dir=${LOG_DIR}

# Training #1 (on train_aug) ###################################################
if [ ${RUN_TRAIN} -eq 1 ]; then
  LIST_DIR=${EXP}/list${LIST_SUFFIX}
  TRAIN_SET=train${TRAIN_SET_SUFFIX}

  TMPFILE_WEAK=$(mktemp /tmp/run_pascal_weak.XXXXXX)
  exec 3>"$TMPFILE_WEAK"
  rm "$TMPFILE_WEAK"

  TMPFILE_STRONG=$(mktemp /tmp/run_pascal_weak.XXXXXX)
  exec 3>"$TMPFILE_STRONG"
  rm "$TMPFILE_STRONG"

  TMPFILE_CMP=$(mktemp /tmp/run_pascal_weak.XXXXXX)
  exec 3>"$TMPFILE_CMP"
  rm "$TMPFILE_CMP"

  grep -o "[0-9]*_[0-9]*.png" ${LIST_DIR}/${TRAIN_SET}.txt        | sort > "$TMPFILE_WEAK"
  grep -o "[0-9]*_[0-9]*.png" ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | sort > "$TMPFILE_STRONG"

  if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
	  TRAIN_SET_WEAK_BBOX=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}

    comm -3 "$TMPFILE_WEAK" "$TMPFILE_STRONG" | sort > "$TMPFILE_CMP"
    ./faster_grep ${TMPFILE_CMP} ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt 
    #grep -f "$TMPFILE_CMP" ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt
  else
	  TRAIN_SET_WEAK_BBOX=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
  
    comm -3 "$TMPFILE_WEAK" "$TMPFILE_STRONG" | head -n ${TRAIN_SET_WEAK_LEN} | sort > "$TMPFILE_CMP"
    ./faster_grep ${TMPFILE_CMP} ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt 
    #grep -f ${TMPFILE_CMP} ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt
  fi
  
  MODEL=${EXP}/model/${NET_ID}/init.caffemodel

  echo "Training net ${EXP}/${NET_ID}"
  for pname in train solver; do
	  sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
  done

  CMD="${CAFFE_BIN} train \
    --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
    --gpu=${GPU_ID}"

	if [ -f ${MODEL} ]; then
	    CMD="${CMD} --weights=${MODEL}"
	fi

	echo Running ${CMD} && ${CMD}
fi

# Test #1 specification (on val or test) #######################################
if [ ${RUN_TEST} -eq 1 ]; then
  for TEST_SET in val; do
	  TEST_ITER=`cat exper/voc12/list${LIST_SUFFIX}/${TEST_SET}.txt | wc -l`
	  MODEL=${EXP}/model/${NET_ID}/test.caffemodel

	  if [ ! -f ${MODEL} ]; then
	      MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
	  fi
	  
	  echo "Testing net ${EXP}/${NET_ID}"
	  FEATURE_DIR=${EXP}/features/${NET_ID}
	  mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
	  mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf

	  sed "$(eval echo $(cat sub.sed))" \
	      ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt

	  CMD="${CAFFE_BIN} test \
      --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
      --weights=${MODEL} \
      --gpu=${GPU_ID} \
      --iterations=${TEST_ITER}"

	  echo Running ${CMD} && ${CMD}
  done
fi

# Training #2 (finetune on trainval_aug) #######################################
if [ ${RUN_TRAIN2} -eq 1 ]; then
  LIST_DIR=${EXP}/list${LIST_SUFFIX}
  TRAIN_SET=trainval${TRAIN_SET_SUFFIX}

  TMPFILE_WEAK=$(mktemp /tmp/run_pascal_weak.XXXXXX)
  exec 3>"$TMPFILE_WEAK"
  rm "$TMPFILE_WEAK"

  TMPFILE_STRONG=$(mktemp /tmp/run_pascal_weak.XXXXXX)
  exec 3>"$TMPFILE_STRONG"
  rm "$TMPFILE_STRONG"

  TMPFILE_CMP=$(mktemp /tmp/run_pascal_weak.XXXXXX)
  exec 3>"$TMPFILE_CMP"
  rm "$TMPFILE_CMP"

  grep -o "[0-9]*_[0-9]*.png" ${LIST_DIR}/${TRAIN_SET}.txt        | sort > "$TMPFILE_WEAK"
  grep -o "[0-9]*_[0-9]*.png" ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | sort > "$TMPFILE_STRONG"

  if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
	  TRAIN_SET_WEAK_BBOX=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}

    comm -3 "$TMPFILE_WEAK" "$TMPFILE_STRONG" | sort > "$TMPFILE_CMP"
    ./faster_grep ${TMPFILE_CMP} ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt 
    #grep -f "$TMPFILE_CMP" ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt
  else
	  TRAIN_SET_WEAK_BBOX=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
  
    comm -3 "$TMPFILE_WEAK" "$TMPFILE_STRONG" | head -n ${TRAIN_SET_WEAK_LEN} | sort > "$TMPFILE_CMP"
    ./faster_grep ${TMPFILE_CMP} ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt 
    #grep -f ${TMPFILE_CMP} ${LIST_DIR}/${TRAIN_SET}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK_BBOX}.txt
  fi

  MODEL=${EXP}/model/${NET_ID}/init2.caffemodel

  if [ ! -f ${MODEL} ]; then
	  MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
  fi
    
  echo "Training2 net ${EXP}/${NET_ID}"
  for pname in train solver2; do
	  sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
  done
  
  CMD="${CAFFE_BIN} train \
    --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
    --weights=${MODEL} \
    --gpu=${GPU_ID}"

	echo Running ${CMD} && ${CMD}
fi

# Test #2 on official test set #################################################
if [ ${RUN_TEST2} -eq 1 ]; then
  for TEST_SET in val; do
	  TEST_ITER=`cat exper/voc12/list${LIST_SUFFIX}/${TEST_SET}.txt | wc -l`
	  MODEL=${EXP}/model/${NET_ID}/test2.caffemodel

	  if [ ! -f ${MODEL} ]; then
	    MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
	  fi
	  
	  echo "Testing2 net ${EXP}/${NET_ID}"
	  FEATURE_DIR=${EXP}/features2/${NET_ID}
	  mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
	  mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
	  sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt

	  CMD="${CAFFE_BIN} test \
      --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
      --weights=${MODEL} \
      --gpu=${GPU_ID} \
      --iterations=${TEST_ITER}"

	  echo Running ${CMD} && ${CMD}
  done
fi
