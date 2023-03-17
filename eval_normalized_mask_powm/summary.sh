#!/bin/bash

subdirs="
normalized_masks_powm_bg-1.0_mode-both_powm-1
normalized_masks_powm_bg-1.0_mode-both_powm-2
normalized_masks_powm_bg-1.0_mode-both_powm-3
normalized_masks_powm_bg-1.0_mode-ideal_powm-1
normalized_masks_powm_bg-1.0_mode-ideal_powm-2
normalized_masks_powm_bg-1.0_mode-ideal_powm-3
normalized_masks_powm_bg-1.0_mode-ideal_powm-4
normalized_masks_powm_bg-1.0_mode-ideal_powm-5
normalized_masks_powm_bg-1.0_mode-left_powm-1
normalized_masks_powm_bg-1.0_mode-left_powm-2
normalized_masks_powm_bg-1.0_mode-left_powm-3
normalized_masks_powm_bg-1.0_mode-right_powm-1
normalized_masks_powm_bg-1.0_mode-right_powm-2
normalized_masks_powm_bg-1.0_mode-right_powm-3

normalized_masks_powm_bg-2.0_mode-both_powm-1
normalized_masks_powm_bg-2.0_mode-both_powm-2
normalized_masks_powm_bg-2.0_mode-both_powm-3
normalized_masks_powm_bg-2.0_mode-ideal_powm-1
normalized_masks_powm_bg-2.0_mode-left_powm-1
normalized_masks_powm_bg-2.0_mode-left_powm-2
normalized_masks_powm_bg-2.0_mode-left_powm-3
normalized_masks_powm_bg-2.0_mode-right_powm-1
normalized_masks_powm_bg-2.0_mode-right_powm-2
normalized_masks_powm_bg-2.0_mode-right_powm-3

normalized_masks_powm_bg-3.0_mode-both_powm-1
normalized_masks_powm_bg-3.0_mode-both_powm-2
normalized_masks_powm_bg-3.0_mode-both_powm-3

normalized_masks_powm_bg-4.0_mode-both_powm-1
normalized_masks_powm_bg-4.0_mode-both_powm-2
normalized_masks_powm_bg-4.0_mode-both_powm-3
normalized_masks_powm_bg-4.0_mode-ideal_powm-1
normalized_masks_powm_bg-4.0_mode-right_powm-1
"

for d in $subdirs; do
    local/write_se_results_as_tsv.sh $d \
    | sed 's/single_mask/dual_masks/g'
done
