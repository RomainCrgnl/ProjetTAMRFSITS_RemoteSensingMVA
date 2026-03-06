# Project 4: Predicting the future to detect changes

## Project architecture
```
.
├── tamrfsits/*        # TAMRF model github repository
├── dataset/*          # Containing the training / test samples
├── model/*            # Containing the pre-trained TAMRFSITS model
├── notebooks/*.ipynb  # Experimental studies
├── src/*.py           # Code for the final solution
├── .gitignore
└── README.md
```

## Installation:
```sh
git clone https://github.com/RomainCrgnl/ProjetTAMRFSITS_RemoteSensingMVA.git
cd ProjetTAMRFSITS_RemoteSensingMVA
```
#### Download training/test data:
Download the dataset: https://zenodo.org/records/15471890

Unzip archives inside the "dataset" folder.


#### Download TAMRF model github repository and dependancies:

This project uses [pixi](https://pixi.sh) as package manager and project configuration tool. Install `pixi` like this for Linux:
```sh
curl -fsSL https://pixi.sh/install.sh | bash
```
or like this for Windows:
```sh
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Clone the `tamrfsits` sources like this:
```sh
git clone https://github.com/Evoland-Land-Monitoring-Evolution/tamrfsits
```

And use `pixi` to install the project and its dependencies:

```sh
cd tamrfsits
pixi install
```

Environment can be activated by using:

```sh
pixi shell
```

#### Download pre-trained model

Pre-trained TAMRFSITS model is available in the following [Zenodo](https://zenodo.org/records/15582231) repository.

> MICHEL, J. (2025). Support data for paper "Temporal Attention Multi-Resolution Fusion of Satellite Image Time-Series, applied to Landsat-8 and Sentinel-2: all bands, any time, at best spatial resolution" (2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17474541

Unzip archives inside the project root directory.