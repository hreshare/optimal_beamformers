#!/usr/bin/env bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:

if [ $# != 1 ]; then
   echo "Wrong #arguments ($#, expected 1)"
   echo "Usage: local/write_se_results.sh <enhancement-method>"
   exit 1;
fi

enhancement=$1

echo -e "PESQ\t$enhancement\tdt05_simu\t$(cat exp/compute_pesq_$enhancement/pesq_dt05)\tet05_simu\t$(cat exp/compute_pesq_$enhancement/pesq_et05)"
echo -e "STOI\t$enhancement\tdt05_simu\t$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_dt05_STOI)\tet05_simu\t$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_et05_STOI)"
echo -e "eSTOI\t$enhancement\tdt05_simu\t$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_dt05_eSTOI)\tet05_simu\t$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_et05_eSTOI)"
echo -e "SDR\t$enhancement\tdt05_simu\t$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_dt05_SDR)\tet05_simu\t$(cat exp/compute_stoi_estoi_sdr_$enhancement/${enhancement}_et05_SDR)"
echo ""
