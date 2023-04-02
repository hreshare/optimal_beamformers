# Mask-based Max SNR, Min NOR, and Max SOR beamformers
This directory includes the experimental system that obtains the optimal mask for each utterance in using the following mask-based beamformers (BFs):

- Maximum Signal-to-Noise Ratio (Max SNR) BF
- Minimum Noise-to-Observation Ratio (Min NOR) BF
- Maximum Signal-to-Observation Ratio (Max SOR) BF



## How to use
1. To make some symbolic links, run the following command just once:
```
    ./prepare.sh
```

2. For the Max SNR BF, run the following commands:
```
    ./run.sh --mode both_max  --bg_gain 1.0
    ./run.sh --mode both_max  --bg_gain 2.0
    ./run.sh --mode both_max  --bg_gain 4.0
```

3. For the Min NOR BF, run the following commands:
```
    ./run.sh --mode left_min  --bg_gain 1.0
    ./run.sh --mode left_min  --bg_gain 2.0
    ./run.sh --mode left_min  --bg_gain 4.0
```

4. For the Max SOR BF, run the following commands:
```
    ./run.sh --mode left_max  --bg_gain 1.0
    ./run.sh --mode left_max  --bg_gain 2.0
    ./run.sh --mode left_max  --bg_gain 4.0
```

5. If you desire the results in the tab-separated value (TSV) format, run the following command;
```
    ./summary.sh | tee results.tsv
```

## Tips
If you would like to confirm the implementation of the mask-based BFs, see the following file:
- local/mask_and_bf/optimal_mormalized_masks.py
