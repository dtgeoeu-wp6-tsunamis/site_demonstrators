## SD5-2 - An Inundation Emulator for the PTF

This repo contains a notebook and instructions to run the SD5 Inundation Emulator for the PTF workflow, and to evaluate the ensemble results.

The main notebook is called ```sd5-2.ipynb```

### Requirements   
The python packages and versions needed to run this notebook are the same needed by the ```inundation-emulator``` module. This was built and tested with python version 3.11.9.   
   
### Inputs

The files required to run the inundation emulator and compare results against the PTF inundation simulations can be obtained from the SDL PTF Experiment (EXP-57 V1.0.0) dataset, along with the emulator model `.keras` file, available at: https://doi.org/10.82554/sdl-57.  The package contains the following:

```bash
EXP-57 V1.0.0/
├── data <!--- ptf simulation time series and inundation netcdf files to copied from here--->
│   ├── PS_Scenario000001_10m.nc
│   ├── PS_Scenario000001_ts.nc
│   ...
│   ├── PS_Scenario001066_10m.nc
│   └── PS_Scenario001066_ts.nc
│   └── scenarios.txt 
├── generated
│   └── emulator_20250516_073408 <!--- emulator model .keras and additional files to be copied from here--->
│       ├── grid_info.json
│       ├── l2_metrics.csv
│       ├── model_checkpoints
│       │   ├── model_epoch_200.keras
│       │   ├── model_epoch_300.keras
│       │   └── model_epoch_50.keras
│       ├── model.keras 
│       ├── plots
│       │   ├── scatter_epoch_100.png
│       │   ├── scatter_epoch_140.png
│       │   ├── scatter_epoch_160.png
│       │   ├── scatter_epoch_180.png
│       │   ├── scatter_epoch_20.png
│       │   ├── scatter_epoch_220.png
│       │   ├── scatter_epoch_240.png
│       │   ├── scatter_epoch_260.png
│       │   ├── scatter_epoch_40.png
│       │   └── scatter_epoch_80.png
│       ├── run.log
│       ├── topography.grd
│       ├── topomask.npy
│       ├── train_summary.csv
│       ├── train_summary_loss.png
│       ├── train_summary_mse.png
│       └── validation_scenarios.txt
└── README
```
Add these files to the project directory of the ```inundation-emulator``` module as shown below. Update any paths in scripts such as `create_emulator.py` and `predict.py` to match your local setup.


```bash
├── create_emulator.py <!--- if training with your own data update all paths  GENERATED_DIR, TRAIN_DIR, VALIDATION_DIR and TRAIN_SCENARIOS match your local directory structure before training --->
├── generated
│   ├── emulator_20250516_073408 <!--- emulator model .keras files are to be added here--->
│   ├── logs
│   └── predictions
├── load_data.py
├── Makefile
├── mask.png
├── notebooks
│   ├── _plots <!--- results and plots from this sd5-2.ipynb notebook are generated here--->
├── pathdirectory.txt
├── poetry.lock
├── predict.py <!--- update all files and paths  GENERATED_DIR, RUN_DIR, PREDICTION_DIR, TEST_DIR and TEST_SCENARIOS match your local directory structure before starting prediction --->
├── pyproject.toml
├── README.md
├── scenario.txt
├── src
│   ├── data_reader.py
│   ├── emulator.py
│   ├── logger_setup.py
│   ├── plotter.py
├── tests
│   ├── data <!--- ptf simulation time series and inundation netcdf files are to be added here--->
│   ├── readme.md
│   └── test_data_reader.py
└── train.txt
```

### Command 
To run the emulator after adding input files and updating paths in `predict.py`:
```terminal
make predictions
```
**Important:** All file paths referenced in the predict.py script (e.g., `GENERATED_DIR`, `TRAIN_SCENARIOS`, `VALIDATION_DIR`, `TOPO_FILE`, `TEST_DIR`, `RUN_DIR`, etc.) must be updated to match your local directory structure before running. For additional help refer ./inundation-emulator/README.md

### Notes
The event ensemble in this work was selected using the PTF workflow from DT-GEO workflow registry (https://gitlab.com/dtgeo/workflow-management-system/workflow-registry/-/tree/main/DTC61/WF6101?ref_type=heads). The tsunami simulations were modelled using Tsunami-HySEA v3.9.0 Monte Carlo edition.



