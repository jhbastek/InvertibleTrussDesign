import pathlib
import numpy as np
import pandas as pd
from src.lattice_utils import Topology
from src.errorAnalysis import compute_NMSE

## define lattice based on output of main_predict.py
# sample no. (leave at 1 if only single stiffness has been passed)
sample = 1
# n-th best prediction of selected sample
pred = 1
# set to true for plotting
plot_lattice = True
# set to true for exporting nodes and connectivities
export_lattice = True

print('Export predicted truss.\n-------------------------------------')

# create directory
pathlib.Path('prediction/conn_coord').mkdir(exist_ok=True)

# load predicted lattice and corresponding target & predicted stiffness
lattice_descriptors = pd.read_csv("prediction/full_pred.csv")
C_target = pd.read_csv("prediction/C_target.csv")
C_pred = pd.read_csv("prediction/C_target_pred_pred.csv")
lattice = lattice_descriptors.loc[lattice_descriptors['sample'] == sample].iloc[pred-1]
C_target = C_target.loc[C_target['sample'] == sample].iloc[0]
C_pred = C_pred.loc[C_pred['sample'] == sample].iloc[pred-1]

# assemble lattice based on descriptor
print('Generate truss connectivities and node coordinates.\n-------------------------------------')
exported_lattice = Topology(lattice)

# print relative density and diameter
if lattice['relative_density'] > 0.1*0.9:
    print('Warning: Relative density approaches upper bound of design space. Try using a stiffer base material for better results.')
elif lattice['relative_density'] < 0.002*1.1:
    print('Warning: Relative density approaches lower bound of design space. Try using a softer base material for better results.')
print('Diameter of (circular) beam cross-section: {:.4f}.\n-------------------------------------'.format(exported_lattice.diameter))

# stiffness evaluation
C_comp = pd.concat([C_target,C_pred],axis=1)[1:]
C_comp.columns = ['Target','Pred.']
print('Stiffness evaluation:\n', C_comp)
print('NMSE: {:.3f}\n-------------------------------------'.format(compute_NMSE(np.expand_dims(C_target[1:].to_numpy()\
    ,axis=0),np.expand_dims(C_pred[1:].to_numpy(),axis=0))[0]))

# plot and export
if plot_lattice:
    exported_lattice.plot()
if export_lattice:
    print('Exporte connectivities and coordinates of lattice.\n-------------------------------------')
    conn=np.array(exported_lattice.connectity)
    coord=np.array(exported_lattice.coordinates)
    np.savetxt('prediction/conn_coord/connectivities.csv', conn, fmt='%i', delimiter=',')
    np.savetxt('prediction/conn_coord/coordinates.csv', coord, delimiter=',')