# Background

Code associated with the research paper Kapp et al., "Data-driven Authoring of Large-scale Ecosystems" 

Some more explanatory comments and minor refactorings (to make the code clearer to read) will be provided on an on-going basis.

Please note that all install instructions were created for Linux distributions. Mac OSX will possibly also work with the instructions.

# Interface install instructions

Install instructions for the interface are in EcoSynth/README.txt.

# Neural network install instructions

The neural networks required to run the authoring tool must be downloaded from https://drive.google.com/file/d/11hEJSL0Ub-yxOZATfqDWEh3Ns9v6GEtS/view?usp=sharing as a zip file. Extract the zip file into your home directory, where it should create a directory called 'ecolearn-models'.

# Project structure

The project is divided into the folders in this directory. A list of the folders, along with a quick summary of the contents, is below:

## EcoSynth

The user interface, which also includes the integration of all backend components, such as the IPC between the Tensorflow session and the interface, the canopy placement, species assignment and control, and undergrowth sampling and refinement.

## UnderSim

The undergrowth simulation model

## canopy\_placement

Canopy placement algorithm. This folder includes all rendering and GPU code necessary for our implementation of the algorithm.

## cluster\_distribs

Undergrowth data analysis, sampling, refinement, as well as some helper code for reading/writing cluster statistics from/to data files.

## gpusample

Contains GPU code for accelerated undergrowth sampling. Used in conjunction with the cluster\_distribs code

## common

Contains some common functions and types used in many different areas of the codebase

## data\_importer

Functions and types used for reading/writing data from/to files

## data\_preproc

Contains programs/projects used for creating datasets ready to use in the interface

## grass\_sim\_standalone

Grass simulator separate from the interface. Useful if grass simulation was not done or written out in interface, but required
afterwards on an output dataset

## species\_optim

Core code used for species assignment, and optimisation based on a brushstroke specifying required species percentages

## tests

Test code. Not nearly extensive enough at the moment - can use a lot more tests

