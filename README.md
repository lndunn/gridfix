# failuremodes

This repository contains the code used in the following paper:

"Bayesian Hierarchical Methods for Modeling Electrical Grid Component Failures" Laurel N. Dunn, Ioanna Kavvada, Mathilde Badoual, Scott J. Moura
[arxiv link](https://arxiv.org/abs/2001.07597)


## User Guide

To use the functions, clone the repository:

```
git clone https://github.com/lndunn/failuremodes.git
```

And install the required modules in a virtual environment:

```
cd failuremodes
pip install virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### failuremodes/ 
Contains the main functions used in the notebooks

### raw_data/
Data used to generate fake data stored in input

### inputs/
Generated fake data that are at the input of the algorithm

### Jupyter Notebooks
Contains the entire data process using the functions in failuremodes/
