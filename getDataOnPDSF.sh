#!/bin/bash

DATA_PATH=/projecta/projectdirs/lux/shared-analysis/detector-perf/data

mkdir -p data

cp ${DATA_PATH}/C14_Run04_peaks.npz data/.
cp ${DATA_PATH}/CH3T_Run04_peaks.npz data/.
cp ${DATA_PATH}/DD_Run04_peaks_*.npz data/.
