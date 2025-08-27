# Generate the database of landslide - tsunami scenarios

This README gives instructions on how to generate a database of landslide-tsunami scenarios for the Messina region.    

The workflow to generate the database follows these 3 steps:
- **STEP 1**: generate the ensemble of landslide release volumes and compute the probabilities of release based on factor of safety and propagation of the release volumes
- **STEP 2**: perform the numerical simulations of landslide dynamics and tsunami wave propagation
- **STEP 3**: process the output of the tsunami simulations to compute the Maximum Inundation Height (MIH)

This pre-computed database is used in the workflow that embeds the landslide sources in the PTF. An example of the database folder with the outputs needed to run the PTF workflow is the input folder of SD5/landslide-workflow.

## STEP 1: Ensemble of representative release volumes
The code for this step is in `release-volume-sampler`.   

This step subdivides the bathymetry in triangles, sets seed triangles based on factor-of-safety, and generates an ensemble of representative release volumes. This is done in the "preparational step". More detailed info can be found in the README of that repo.

A jupiter notebook to run this step is provided in `release-volume-sampler/notebooks/preparational.ipynb`.    

Parameters associated with the propagation of release volumes and selection of seed triangles can be changed to obtain different volumes ensembles. 

The main outputs of this step are:
- *out-1.1*: a .csv file with the list of landslide release volumes and associated parameters (e.g., probabilities, area, volume
, coordinates of the computational domain, ... )
- *out-1.2*: an ascii file for each release volume with the thickness of the slide and coordinates in the format readable by BingClaw (for STEP 2) 
- a volume database (.db) that contains all the info of each release volumes. This file is not used here, but it will be used in the "operational step" when the database is used in connection to the PTF.
- other miscellaneous files describing the triangulation and slope analysis.

## STEP 2: Landslide and tsunami simulations
The code for this step is in `bingclaw-to-hysea`.   

This step runs the landslide simulations with the code BingClaw, runs the interface module that creates a sea floor deformation file from the output of BingClaw output that can be used by Tsunami-HySEA, and runs the tsunami simulations with the code Tsunami-HySEA. More detailed info, especially regarding requirements and inputs, can be found in the README of that repo.   

The main script to run this part of the workflow is `bingclaw-to-hysea/read_volumes_create_runscripts.py`. This script reads *out-1.1* and, for each scenario, launches the BingClaw simulation (which uses *out-1.2*), the interface module, and T-HySEA.   

The main outputs of this step are:
- Output of the BingClaw simulations for each scenario. This includes files with the thickness, velocity, and location of the slide at each time step.
- Output of the T-HySEA simulations for each scenario. This include a file with the maximum wave height, eta, ... and a file (*out-2.1*) with the time series describing the wave forms for each Point of Interest. 

## STEP 3: Compute the Maximum Inundation Height (MIH)
The code for this step is in `landslide-wf-postproc`.   

This step extracts peaks from the wave forms of the tsunami simulations and uses the amplification factor method to compute the MIH values at each POI of each scenario. More detailed info can be found in the README of that repo.   

The main script to run this step is `run_postproc_all.py`. This script reads *out-1.1* and uses *out-2.1* to compute the MIHs.   

The main output of this step is: 
- a .npy file with a 2D array with shape containing the MIH values for each POI for each scenario.   

This is the final output of the workflow, which is needed when this pre-computed database is used in the workflow linked to the PTF.
