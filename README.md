# Inverting the structury-property map of truss metamaterials via deep learning

We introduce a framework to inversely design truss metamaterials with a given anisotropic elasticity as described in ['Inverting the structure-property map of truss metamaterials via deep learning'](https://www.pnas.org/content/119/1/e2111505119). 
This repository consists of three main scripts:
1. **main_train.py** trains the presented framework with lattice-stiffness pairs computed with an inhouse finite element simulation. It should only be used if one is interested to retrain the networks, e.g., to reproduce the evaluation presented in the paper. The dataset can be found under this [link](https://www.research-collection.ethz.ch/handle/20.500.11850/520254).
2. **main_predict.py** predicts and stores a variety of inverse designs given a certain set of anistropic stiffness tensors. It must be provided in a .csv-file, as, e.g., the provided anisotropic bone samples in 'data/prediction.csv' (which can be run to reproduce the presented results). Note that the predicted stiffnesses slightly differ from the ones presented in the paper, which were computed and verified using our inhouse finite element framework.
3. **main_export.py** plots the predicted lattice descriptor and converts it into a list of nodal position and connectivities for further postprocessing. Additionally, it warns the user if the requested stiffness is to stiff or soft for the considered range of relative densities and Young's modulus, in which case main_predict.py should be rerun with a suitable Young's modulus of the base material.

For further information, please first refer to the [paper](https://www.pnas.org/content/119/1/e2111505119), [supporting information](https://www.pnas.org/content/pnas/suppl/2021/12/30/2111505119.DCSupplemental/pnas.2111505119.sapp.pdf), or reach out to [Jan-Hendrik Bastek](mailto:jbastek@ethz.ch).

## Requirements

- Python (tested on version  3.7.1)
- Python packages:
  - Pytorch (tested on CUDA version 11.4)
  - Pandas
  - NumPy
  - Plotly (used to plot the inversely designed structures)
