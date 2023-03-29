#!/bin/bash

subdirs="
mask_transfer_bg-1.0_from-left_max_to-right_powm-1
mask_transfer_bg-2.0_from-left_max_to-right_powm-1
mask_transfer_bg-4.0_from-left_max_to-right_powm-1

mask_transfer_bg-1.0_from-left_min_to-right_powm-1
mask_transfer_bg-2.0_from-left_min_to-right_powm-1
mask_transfer_bg-4.0_from-left_min_to-right_powm-1
"

for d in $subdirs; do
    local/write_se_results_as_tsv.sh $d \
    | sed 's/single_mask/dual_masks/g'
done
