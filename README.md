# Site Demonstrators
This repository contains all the codes and jupiter notebooks to run the site demonstrator test cases of WP6 for the DT-GEO project (deliverable 6.6).

Each SD directory contains a notebook that runs all the relevant steps of the workflow for that specific site demonstrator.

## SD4 - Mediterranean Sea
The notebook `SD4/sd4.ipynb` executes an end-to-end execution of the Probabilistic Tsunami Forecasting (PTF) workflow in its basic capabilities for the October 30, 2020, Mw 7.0 Samos-Izmir earthquake.  

## SD5 - East Sicily
This site demonstrator has been used to showcase two new functionalities added to the PTF workflow. 

### SD5-1: Embedding landslide tsunami sources into the PTF
The notebook `SD5/landslide-workflow/sd5-1.ipynb` uses the December 28,1908 Mw of 7.1 Messina event as a test case for the mini-workflow that embeds landslide sources into the PTF. It shows how the pre-computed database of landslide and tsunami simulations is used to compute the probabilities of maximum inundation heights along the coast of East Sicily, given the ensemble of earthquake sources compute by the first step of the PTF.

### SD5-2: Inundation emulator for the PTF
The notebook `SD5/inundation-emulator/sd5-2.ipynb` shows how to use the AI inundation emulator for East Sicily given the PTF ensemble of a synthetic event of Mw 8.05 along the Calabrian subduction zone.

## SD6 - Chile event and data assimilation
The notebook `SD6/sd6.ipynb` executes an end-to-end execution of the global version of PTF workflow for the February 27, 2010, Maule (Chile) Mw 8.8 earthquake. This example includes the execution of the misfit evaluator module, which is a new functionality added to the PTF for assimilation of sea level data.

## SD7 - Japan event
The notebook `SD7/sd7.ipynb` executes an end-to-end execution of the global version of PTF workflow for the March 11, 2011, Mw = 9.1 Tohoku (Japan) earthquake. This example shows how the PTF performs with an earthquake that has a complex source with an atypical seismic rupture pattern.

## License
The content of this repository is licensed under the Non-Profit Open Software License version 3.0 (NPOSL-3.0).  
See the [LICENSE](./LICENSE.txt) file for details.