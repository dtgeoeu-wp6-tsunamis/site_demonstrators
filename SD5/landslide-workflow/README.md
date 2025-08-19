## SD5 - 1908 Messina event: a test case for adding landslide sources to the PTF   

The `SD5/landslide-workflow` folder contains a notebook and codes needed to run the Site Demonstrator SD5 1908 Messina event to show the embedding of the landslide scenarios in the PTF workflow.   

This folder contains:
- `sd5-1.ipynb`: main notebook to run the SD5 example
- `ewricagm`: code for computing shakemaps
- `precomputed-database`: 
    - `release-volume-sampler`: code for creating the list of release volumes (input for the landslide simulations) and to compute the probabilities given the shakemaps and the probabilities of the PTF scenarios
    - `bingclaw-to-hysea`: code to run the landslide dynamics simualations and the tsunami simulations
    - `landslide-wf-postproc`: code to compute the maximum inundation heights from the HySEA simulations using the amplification factors and to plot the results
- `utils`: functions needed to run the sd5-1 notebook 
- `output`: folder where the output of the notebook are stored (it is created at runtime if not present yet)


### Instructions for the use of submodules
After cloning this repository, it is necessary to run an additional command to download also the repositories stored as submodules:
```
git submodule update --init --recursive
```
If any changes to the original repositories are made, the following command is to align the submodules to the latest commit in the original repository:

```
git submodule update --remote --merge
```

### Requirements   
The file `requirements.txt` contains all the python packages and versions needed to run this notebook. This was tested with python version 3.11.8.   
Note that some parts of the workflow require very specific versions of python packages (e.g., tensorflow 2.12) and specific wheels, so we recommend to use the `requirements.txt` file to install the right versions of the python packages with pip:
```
pip install -r requirements.txt
```
   
### Inputs
The files needed to run this notebook can be downloaded from ADD LINK.    
The input folder has the following structure:   
```
input-landslide-workflow-sd5
| - precomputed-database
|   | - ...
| - ptf-input
|   | - FocMech_PreProc
|   | - mesh_files
```   

### Acknowledgments 
The original PTF code is available from the DT-GEO workflow registry (https://gitlab.com/dtgeo/workflow-management-system/workflow-registry/-/tree/main/DTC61/WF6101?ref_type=heads). For the sake of simplicity and conciseness, the codes in this notebook (as well in the external functions.ipynb notebook) have been revised and some general options have been removed, to focus on the specific site demonstrator.   
   
The code to compute the shakemaps is based on Lucas Lehmann's code (https://github.com/rauler95/gmacc) end EWRICA project (https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=79057f75-b358-11ed-95b8-f851ad6d1e4b). The application of Lehmann's code is described in the following publication: https://academic.oup.com/gji/article/234/3/2328/7202317.   
   
### License
This project is licensed under the Non-Profit Open Software License version 3.0 (NPOSL-3.0).  
See the [LICENSE](./LICENSE) file for details.


