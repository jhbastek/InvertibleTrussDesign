import numpy as np
import pandas as pd
# from lattice_utils import Topology

## define lattice to plot
sample = 1
pred = 1

C_target = pd.read_csv("Predictions/C_target.csv")
lattice_descriptors = pd.read_csv("Predictions/full_pred.csv")
C_pred = pd.read_csv("Predictions/C_target_pred_pred.csv")

C_target = C_target.loc[C_target['sample'] == sample].iloc[pred-1]
lattice = lattice_descriptors.loc[lattice_descriptors['sample'] == sample].iloc[pred-1]
C_pred = C_pred.loc[C_pred['sample'] == sample].iloc[pred-1]

print(lattice)
# Topology(lattice)