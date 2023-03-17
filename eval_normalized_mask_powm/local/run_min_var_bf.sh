#!/usr/bin/env bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

if [ $# != 7 ]; then
   echo "Wrong #arguments ($#, expected 7)"
   echo "Usage: local/run_train_and_infer.sh <wav-in-dir> <wav-out-dir> <track> <bg_gain> <mode> <do_plot> <powm_itr>"
   exit 1;
fi

sdir=$1
odir=$2
track=$3
bg_gain=$4
mode=$5
do_plot=$6
powm_itr=$7
max_epochs=500

#gpu_id=1
gpu_id=-1
case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac 

##if [ ! -f local/nn-gev/data/BLSTM_model/mlp.tr ]; then
#if [ ! -f local/mask_and_bf/data/BLSTM_model/best.nnet ]; then
#    echo "training a BLSTM mask network"
#    #$HOME/miniconda3/bin/python local/nn-gev/train.py --chime_dir=$sdir/data --gpu $gpu_id local/nn-gev/data BLSTM
#    #python3 local/mask_and_bf/train.py --chime_dir=$sdir/data --gpu $gpu_id local/mask_and_bf/data BLSTM
#    python3 local/mask_and_bf/train.py \
#      --chime_dir=$sdir/data \
#      --gpu $gpu_id \
#      local/mask_and_bf/data \
#      BLSTM
#
##      --max_epochs 20 \
##      --patience -1 \
#
#else
#    echo "Not training a BLSTM mask network. Using existing model in local/mask_and_bf/data/BLSTM_model/"
#fi

echo "enhancing signals with weighted minimum variance beamformer"
local/mask_and_bf/beamform.sh \
  $sdir/data \
  local/mask_and_bf/data \
  $odir \
  --gpu $gpu_id \
  --track $track \
  --mode $mode \
  --bg_gain $bg_gain \
  --plot $do_plot \
  --max_epochs $max_epochs \
  --powm_itr $powm_itr \
  

