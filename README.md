# optimal_beamformers
This repository includes systems for exploring peak performance of the following mask-based beamformers (BFs):
- Maximum Signal-to-Noise Ratio (Max SNR) BF
- Minimum Noise-to-Observation Ratio (Min NOR) BF
- Maximum Signal-to-Observation Ratio (Max SOR) BF
- Multichannel Wiener filter (MWF) BF


Experimental systems are classified as follows:
1. Obtaining the optimal mask for each utterence
   - For Max SNR, Min NOR, and Max SOR BFs
     - eval_normalized_mask_eigh/
   - For MWF
     - eval_normalized_mask_powm/

2. Applying the optimal mask for a BF method to another
   - MWF --> Max SNR, Min NOR, and Max SOR
     - eval_mask_transfer/
   - Max SNR, Min NOR, and Max SOR --> Max SNR, Min NOR, and Max SOR
     - eval_mask_transfer_eigh_to_eigh/
   - Max SNR, Min NOR, and Max SOR --> MWF
     - eval_mask_transfer_eigh_to_mwf/


## How to use
These systems use the CHiME 4 dataset and depend on its baseline system for this dataset included in the Kaldi Speech Recognition Toolkit. Thus, the following steps are necesarry before using the experimental systems:

1. Obtain the CHiME 4 dataset. See https://spandh.dcs.shef.ac.uk/chime_challenge/chime2016/index.html .
2. Set up the Kaldi Speech Recognition Toolkit. See https://github.com/kaldi-asr/kaldi .
3. In that toolkit, go to kaldi/egs/chime4/s5_6ch/ and conduct the experiments using the baseline system.

After that, do following steps:

4. Copy the following directories to kaldi/egs/chime4/:
   - eval_normalized_mask_eigh/
   - eval_normalized_mask_powm/
   - eval_mask_transfer/
   - eval_mask_transfer_eigh_to_eigh/
   - eval_mask_transfer_eigh_to_mwf/
5. Go to each directory to conduct the experiment there.
