# Band extraction for the LUX EFT andlysis

Contact: Q.Riffard (qriffard -at- lbl.gov)
--
This repository is about the extraction of the band for the calibration of the LUX EFT analysis.


--

## Usage:

### 1.Get the data

The first step is to get the data that you are going to need for the extraction of the bands. There are two solutions:

- On you laptop: you have to download the data from NERSC by using the macro `getDataFromPDSF.sh`. Usage:
```shell
./getDataFromPDSF.sh <username>
```

- On PDSF: you have don't have to copy the data, a simlink to the data folder is perfect:
```shell
ln -s /global/projecta/projectdirs/lux/shared-analysis/detector-perf/data .
```

These two macros are going to pull the Run4 data and create a directory called `data` 

### 2. Run the band extraction

To run the band extraction, execute all the cells of the notebook. The output will be stored in the directory `bands`


--
## Acknolegment

Spetial thank to Vetri and Matthew for their help about this work.