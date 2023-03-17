#!/bin/bash -x

#!/usr/bin/env bash

#for flist in tr05_simu tr05_real dt05_simu dt05_real et05_simu et05_real; do
#     python3 local/mask_and_bf/beamform.py $flist "$@"
#done
for flist in dt05_simu et05_simu ; do
     python3 local/mask_and_bf/optimal_mormalized_masks.py $flist "$@"
     #python3 local/mask_and_bf/optimal_bf_mask.py $flist "$@"
done
