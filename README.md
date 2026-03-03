# Project 4: Predicting the future to detect changes

## Installation:
```sh
git clone https://github.com/RomainCrgnl/ProjetTAMRFSITS_RemoteSensingMVA.git
cd ProjetTAMRFSITS_RemoteSensingMVA
```
#### Download training/test data:
Download the dataset: https://zenodo.org/records/15471890

Unzip archives inside the "dataset" folder.


#### Download TAMRF model github repository and dependancies:

This project uses [pixi](https://pixi.sh) as package manager and project configuration tool. Install `pixi` like this:
```sh
curl -fsSL https://pixi.sh/install.sh | bash
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


## Project architecture
```
.
├── tamrfsits/*        # TAMRF model github repository
├── dataset/*          # Containing the training / test samples
├── notebooks/*.ipynb  # Experimental studies
├── src/*.py           # Code for the final solution
├── .gitignore
└── README.md
```