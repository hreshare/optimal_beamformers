# Mask application across BF methods
This directory includes the experimental system that applies the optimal mask for each utterance obtained in using the multichannel Wiener filter beamformer (MWF BF) to the following BFs:

- Minimum Noise-to-Observation Ratio (Min NOR) BF
- Maximum Signal-to-Observation Ratio (Max SOR) BF



## How to use
1. To obtain and store the optimal mask for each utterance, run the experiments in the following directories:
    - eval_normalized_masks_eigh
    - eval_normalized_masks_powm

2. To make some symbolic links, run the following command just once:
```
    ./prepare.sh
```

3. For applying the optimal mask to the Min NOR BF, run the following commands:
```
    ./run.sh --mode eig_left_min --bg_gain 1.0
    ./run.sh --mode eig_left_min --bg_gain 2.0
    ./run.sh --mode eig_left_min --bg_gain 4.0
```

4. For applying the optimal mask to the Max SOR BF, run the following commands:
```
    ./run.sh --mode eig_left_max --bg_gain 1.0
    ./run.sh --mode eig_left_max --bg_gain 2.0
    ./run.sh --mode eig_left_max --bg_gain 4.0
```

5. If you desire the results in the tab-separated value format, run the following command;
```
    ./summary.sh
```

## Tips
If you would like to confirm the implementation of the mask-based BFs, see the following file:
- local/mask_and_bf/optimal_mormalized_masks.py
