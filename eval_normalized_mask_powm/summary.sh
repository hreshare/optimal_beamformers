#!/bin/bash

subdirs="
normalized_masks_powm_bg-1.0_mode-right_powm-1
normalized_masks_powm_bg-2.0_mode-right_powm-1
normalized_masks_powm_bg-4.0_mode-right_powm-1

normalized_masks_powm_bg-1.0_mode-ideal_powm-1
normalized_masks_powm_bg-2.0_mode-ideal_powm-1
normalized_masks_powm_bg-4.0_mode-ideal_powm-1
"

for d in $subdirs; do
    local/write_se_results_as_tsv.sh $d \
    | sed 's/single_mask/dual_masks/g'
done
