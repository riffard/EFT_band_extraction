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

- On PDSF: you have to copy a local version of the data by using the macro `getDataOnPDSF.sh`. The space occupiued by the data is approximatively 2Go.
Usage:
```shell
./getDataOnPDSF.sh
```

These two macros are going to pull the Run4 data and create a directory called `data` 

### 2. Run the band extraction




--
## Acknolegment

Spetial thank to Vetri and Matthew for their help about this work.