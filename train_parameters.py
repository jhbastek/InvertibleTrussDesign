import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training data path
dataPath = 'data/training.csv'

# define column names from data
F1_features_names = ['relative_density','U1','U2','U3','lattice_type1','lattice_type2','lattice_type3','lattice_rep1','lattice_rep2','lattice_rep3']
R1_names = ['R1_theta','R1_rot_ax1','R1_rot_ax2']
V_names = ['V1','V2','V3']
R2_names = ['R2_theta','R2_rot_ax1','R2_rot_ax2']
all_names = F1_features_names + R1_names + R2_names + V_names
C_ort_names = ['C11_ort','C12_ort','C13_ort','C22_ort','C23_ort','C33_ort','C44_ort','C55_ort','C66_ort']
C_names = ['C11','C12','C13','C14','C15','C16','C22','C23','C24','C25','C26','C33','C34','C35','C36','C44','C45','C46','C55','C56','C66']

# define data type for correct scaling
F1_features_types = ['continuous']*4 + ['categorical']*6
V_types = ['continuous']*3
C_ort_types = ['continuous']*9
C_types = ['continuous']*21

# define scaling strategies
F1_features_scaling_strategy = 'min-max-1'
V_scaling_strategy = 'min-max-1'
C_ort_scaling_strategy = 'min-max-2'
C_scaling_strategy = 'min-max-2'
C_hat_scaling_strategy = 'min-max-2'

# define train/test split
traintest_split = 0.99

# define batch size and numWorkers
batchSize = 8192
numWorkers = 0

## define training protocol
# F1 training
F1_train = False
F1_train_epochs = 100
F1_arch = [512,'leaky',512,'leaky']
F1_learning_rate = 1e-3
# F2 training
F2_train = False
F2_train_epochs = 150
F2_arch = [1024,'leaky',1024,'leaky',1024,'leaky']
F2_learning_rate = 1e-3
# inverse training (G1 & G2)
inv_train = False
inv_train_epochs = 250
inv_arch = [2560,'leaky',2560,'leaky',1280,'leaky',1280,'leaky',640,'leaky',640,'leaky']
inv_learning_rate = 1e-4

# define loss function
lossFn = torch.nn.MSELoss()