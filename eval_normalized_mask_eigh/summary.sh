#!/bin/bash

subdirs="
normalized_masks_eigh_bg-1.0_mode-both_max
normalized_masks_eigh_bg-2.0_mode-both_max
normalized_masks_eigh_bg-4.0_mode-both_max

normalized_masks_eigh_bg-1.0_mode-left_min
normalized_masks_eigh_bg-2.0_mode-left_min
normalized_masks_eigh_bg-4.0_mode-left_min

normalized_masks_eigh_bg-1.0_mode-left_max
normalized_masks_eigh_bg-2.0_mode-left_max
normalized_masks_eigh_bg-4.0_mode-left_max
"


for d in $subdirs; do
    local/write_se_results_as_tsv.sh $d \
    | sed 's/single_mask/dual_masks/g'
done
