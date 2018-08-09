#!/bin/bash

DATA_PATH=/projecta/projectdirs/lux/shared-analysis/detector-perf/data
USER=${1}

if [[ -z "$USER" ]];then
    echo "Wrong usage: ./getDataFromPDSF.sh user_name"
    exit
fi

mkdir -p data

rsync -avz --progress ${USER}@pdsf.nersc.gov:${DATA_PATH}/C14_Run04_peaks.npz data/.
rsync -avz --progress ${USER}@pdsf.nersc.gov:${DATA_PATH}/CH3T_Run04_peaks.npz data/.
rsync -avz --progress ${USER}@pdsf.nersc.gov:${DATA_PATH}/DD_Run04_peaks_*.npz data/.
