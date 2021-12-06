# Inverse design of metamaterials with given anisotropic stiffness

We introduce a framework to inversely design metamaterials with a given anisotropic elasticity, as presented in our paper 'Inverting the structure-property map of truss metamaterials via deep learning'. It consists of three main scripts:
1) main_train.py trains the presented framework with lattice-stiffness pairs computed with an inhouse FEM simulation. It should only be used if one is interested to retrain the networks, e.g., to verify the evaluation presented in the paper. The dataset can be found here:
2) main_predict.py predicts and stores a variety of inverse designs given a certain anistropic stiffness tensor. It must be provided in a .csv-file, as, e.g., the anisotropic bone samples in 'data/pred_data.csv'.
3) main_export.py plots the predicted lattice descriptor and converts it into a list of nodal position and connectivities for further postprocessing. Additionally, it warns the user if the requested stiffness is to stiff or soft for the considered range of relative densities and Young's modulus, in which case main_predict.py should be rerun with a suitable Young's modulus. 

Tested using: Pytorch, Pandas, Numpy, Plotly

For more information, please refer to the following:

  - Bastek, Jan-Hendrik; Kumar, Siddhant; Telgen, Bastian; Glaesener, Raphael and Kochmann, Dennis M."[Inverting the structure-property map of truss metamaterials via deep learning](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Proceedings of the National Academy of Sciences (PNAS).