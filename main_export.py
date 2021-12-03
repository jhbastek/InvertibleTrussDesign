import pathlib
import numpy as np
import pandas as pd
from lattice_utils import Topology

## define lattice based on predictions
# sample
sample = 1
# n-th best prediction of sample
pred = 1

# set to true if 
plot_lattice = True
# set to true if 
export_lattice = True

# create directory
pathlib.Path('predictions').mkdir(exist_ok=True)
pathlib.Path('predictions/conn_coord').mkdir(exist_ok=True)

# load predicted lattice and corresponding target + predicted stiffness
C_target = pd.read_csv("Predictions/C_target.csv")
lattice_descriptors = pd.read_csv("Predictions/full_pred2.csv")
C_pred = pd.read_csv("Predictions/C_target_pred_pred.csv")
C_target = C_target.loc[C_target['sample'] == sample].iloc[pred-1]
lattice = lattice_descriptors.loc[lattice_descriptors['sample'] == sample].iloc[pred-1]
C_pred = C_pred.loc[C_pred['sample'] == sample].iloc[pred-1]

# assemble lattice based on descriptor
exported_lattice = Topology(lattice)

print('Diameter of beam cross-section: ', exported_lattice.diameter)
# print(C_target)

if plot_lattice:
    exported_lattice.plot()

if export_lattice:
    print('Export connectivities and coordinates of lattice.')
    conn=np.array(exported_lattice.connectity)
    coord=np.array(exported_lattice.coordinates)
    np.savetxt('predictions/conn_coord/connectivity.csv', conn, fmt='%i', delimiter=',')
    np.savetxt('predictions/conn_coord/coordinates.csv', coord, delimiter=',')