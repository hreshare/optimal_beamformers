# Mask application across BF methods
This directory includes the experimental system that applies the optimal mask obtained using one of the following to the other:

- Minimum Noise-to-Observation Ratio (Min NOR) BF
- Maximum Signal-to-Observation Ratio (Max SOR) BF



## How to use
1. To obtain and store the optimal mask for each utterance, conduct the experiments in the following directories:
    - eval_normalized_masks_eigh
    - eval_normalized_masks_powm

2. To make some symbolic links, run the following command just once:
```
    ./prepare.sh
```

3. For applying the optimal mask obtained using the Max SOR BF to the Min NOR BF, run the following commands:
```
    ./run.sh --bg_gain 1.0 --mode_from left_max --mode_to left_min
    ./run.sh --bg_gain 2.0 --mode_from left_max --mode_to left_min
    ./run.sh --bg_gain 4.0 --mode_from left_max --mode_to left_min
```

4. For applying the optimal mask obtained using the Min NOR BF to the Max SOR BF, run the following commands:
```
    ./run.sh --bg_gain 1.0 --mode_from left_min --mode_to left_max
    ./run.sh --bg_gain 2.0 --mode_from left_min --mode_to left_max
    ./run.sh --bg_gain 4.0 --mode_from left_min --mode_to left_max
```

5. If you desire the results in the tab-separated value (TSV) format, run the following command;
```
    ./summary.sh | tee results.tsv
```

## Tips
If you would like to confirm the implementation of the mask-based BFs, see the following file:
- local/mask_and_bf/optimal_mormalized_masks.py
