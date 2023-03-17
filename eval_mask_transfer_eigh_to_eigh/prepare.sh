#!/bin/bash -x

ln -s ../s5_6ch/{rnnlm,steps,utils,conf} .

(mkdir -p local/mask_and_bf/data ;\
 cd local/mask_and_bf/data ;\
 ln -s ../../../../s5_6ch/local/nn-gev/data/audio . ;\
)
