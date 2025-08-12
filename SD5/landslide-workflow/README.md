## SD5 - 1908 Messina event: a test case for adding landslide sources to the PTF   

This repo contains a notebook and codes needed to run the SD5 1908 Messina event to show the embedding of the landslide scenarios in the PTF workflow.   

The main notebook is called ```sd5.ipynb```

### Requirements   
The file "requirements.txt" contains all the python packages and versions needed to run this notebook. This was tested with python version 3.11.8.   
   
### Inputs
The files need to run this notebook can be downloaded from ADD LINK

### Acknowledgments 
The original PTF code is available from the DT-GEO workflow registry (https://gitlab.com/dtgeo/workflow-management-system/workflow-registry/-/tree/main/DTC61/WF6101?ref_type=heads). For the sake of simplicity and conciseness, the codes in this notebook (as well in the external functions.ipynb notebook) have been revised and some general options have been removed, to focus on the specific site demonstrator.   
   
The code to compute the shakemaps is based on Lucas Lehmann's code (https://github.com/rauler95/gmacc) end EWRICA project (https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=79057f75-b358-11ed-95b8-f851ad6d1e4b). The application of Lehmann's code is described in the following publication: https://academic.oup.com/gji/article/234/3/2328/7202317.   
   



