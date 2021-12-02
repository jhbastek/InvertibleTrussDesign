import pathlib
import pandas as pd
from lattice_utils import Topology

# create directory
pathlib.Path('predictions').mkdir(exist_ok=True)
pathlib.Path('predictions/conn_coord').mkdir(exist_ok=True)

## define lattice to plot
sample = 2
pred = 2

C_target = pd.read_csv("Predictions/C_target.csv")
lattice_descriptors = pd.read_csv("Predictions/full_pred2.csv")
C_pred = pd.read_csv("Predictions/C_target_pred_pred.csv")

C_target = C_target.loc[C_target['sample'] == sample].iloc[pred-1]
lattice = lattice_descriptors.loc[lattice_descriptors['sample'] == sample].iloc[pred-1]
C_pred = C_pred.loc[C_pred['sample'] == sample].iloc[pred-1]

test = Topology(lattice)

conn = test.connectity
coords = test.coordinates
diameter = test.diameter
print(conn.shape)
print(coords.shape)
print(diameter)

# test.plotTruss()


