Rosenblatt Perceptron & Learning a Linearly Separable Rule
====
_made by: Sebastian Prehn and Niklas Erdmann_

Note: This contains both the code for Assignment 1 and 2. While codefiles for both are distinct, the readme is not.

**Package requirements:**
 - numpy
 - matplotlib
 - pandas
 - seaborn
 - sys
 - ast

(assignment 2 doesn't require sys and ast, but both should be installed by default)

## **Assignment 1 - Rosenblatt Perceptron:**

Run the algorithm via running `python perceptron_training.py` followed by some input arguments which will determine the experiment set-up:

possible arguments are structured like so: `run [nD,nmax,[N]] printEpochs`

`run` will run the experiment and plot the results. It can be replaced by `plot` which will produce only the plot displayed in the report. The data for the plot is hardcoded (saved from a previous result), thus wont be changed by running experiments.  

`[nD,nmax,[N]]` are parameter settings. N is an extra list, as it is possible to pass several N for testing. 

`printEpochs` is a boolean which determines, if in addition to final proportion of successful runs, also average epoch count needed is printed in the terminal.

Here are some example configurations:

`python perceptron_training.py run [50,100,[100]] False`

`python perceptron_training.py run [50,100,[10,20,100]] False`

`python perceptron_training.py plot`

passing no arguments but `run` will run the experiment in standard configuration: `[50,100,[10,20,100]]`


## **Assignment 2 - Learning a Linearly Separable Rule:**

This algorithm is less modular than in assignment 1. It can be executed by simply running `python minOver.py` 

Figures created by this algorithm (histograms of stability distributions and generalization error vs p/n) are saved in the folder `figures_2` which is created (if it does not exist) in your working directory.

