# Mask-based multichannel Wiener filter beamformer
This directory includes the experimental system that obtains the optimal mask for each utterance in using the mask-based and ideal multichannel Wiener filter beamformers (MWF BFs).

## How to use
1. To make some symbolic links, run the following command just once:
```
    ./prepare.sh
```

2. For the mask-based MWF BF, run the following commands:
```
    ./run.sh --mode right  --powm_itr 1  --bg_gain 1.0
    ./run.sh --mode right  --powm_itr 1  --bg_gain 2.0
    ./run.sh --mode right  --powm_itr 1  --bg_gain 4.0
```

3. For the ideal MWF BF, run the following commands:
```
    ./run.sh --mode ideal  --powm_itr 1  --bg_gain 1.0
    ./run.sh --mode ideal  --powm_itr 1  --bg_gain 2.0
    ./run.sh --mode ideal  --powm_itr 1  --bg_gain 4.0
```

4. If you desire the results in the tab-separated value format, run the following command;
```
    ./summary.sh
```

## Tips
If you would like to confirm the implementation of the mask-based BFs, see the following file:
- local/mask_and_bf/optimal_mormalized_masks.py
