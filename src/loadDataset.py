import torch
from torch.utils.data import TensorDataset
import pickle
import numpy as np
import pandas as pd
from train_parameters import *
from src.normalization import Normalization
from src.voigt_rotation import *
from src.model_utils import CPU_Unpickler
  
def exportTensor(name,data,cols, header=True):
    df=pd.DataFrame.from_records(data.detach().numpy())
    if(header):
        df.columns = cols
    print(name)
    df.to_csv(name+".csv", header=header, index=False)

def exportList(name,data):
    arr=np.array(data)
    np.savetxt(name+".csv", [arr], delimiter=',')

def getNormalization(save_normalization=False):
    
    data = pd.read_csv(dataPath)
    # check for NaNs 
    assert not data.isnull().values.any()
    
    F1_features = torch.tensor(data[F1_features_names].values)
    R2 = torch.tensor(data[R2_names].values)
    V = torch.tensor(data[V_names].values)
    C_ort = torch.tensor(data[C_ort_names].values)
    C = torch.tensor(data[C_names].values)

    a,b,c = torch.split(R2,[1,1,1],dim=1)
    R2_transposed = torch.cat((-a,b,c),dim=1)
    unrotatedlabelTensor = direct_rotate(C,R2_transposed)

    F1_features_scaling = Normalization(F1_features,F1_features_types,F1_features_scaling_strategy)
    V_scaling = Normalization(V,V_types,V_scaling_strategy)
    C_ort_scaling = Normalization(C_ort, C_ort_types,C_ort_scaling_strategy)
    C_scaling = Normalization(C,C_types,C_scaling_strategy)
    C_hat_scaling = Normalization(unrotatedlabelTensor,C_types,C_hat_scaling_strategy)

    # should only be activated if framework is retrained with different dataset
    if save_normalization:
        with open('src/normalization/F1_features_scaling.pickle', 'wb') as file_:
            pickle.dump(F1_features_scaling, file_, -1)
        with open('src/normalization/V_scaling.pickle', 'wb') as file_:
            pickle.dump(V_scaling, file_, -1)
        with open('src/normalization/C_ort_scaling.pickle', 'wb') as file_:
            pickle.dump(C_ort_scaling, file_, -1)
        with open('src/normalization/C_scaling.pickle', 'wb') as file_:
            pickle.dump(C_scaling, file_, -1)
        with open('src/normalization/C_hat_scaling.pickle', 'wb') as file_:
            pickle.dump(C_hat_scaling, file_, -1)
    return F1_features_scaling, C_ort_scaling, C_scaling, V_scaling, C_hat_scaling

def getSavedNormalization():
       
    F1_features_scaling = CPU_Unpickler(open("src/normalization/F1_features_scaling.pickle", "rb", -1)).load()
    V_scaling = CPU_Unpickler(open("src/normalization/V_scaling.pickle", "rb", -1)).load()
    C_ort_scaling = CPU_Unpickler(open("src/normalization/C_ort_scaling.pickle", "rb", -1)).load()
    C_scaling = CPU_Unpickler(open("src/normalization/C_scaling.pickle", "rb", -1)).load()
    C_hat_scaling = CPU_Unpickler(open("src/normalization/C_hat_scaling.pickle", "rb", -1)).load()
    return F1_features_scaling, C_ort_scaling, C_scaling, V_scaling, C_hat_scaling

def getDataset(F1_features_scaling, V_scaling, C_ort_scaling, C_scaling):
      
    data = pd.read_csv(dataPath)
    
    print('Data: ',data.shape)       
    # check for NaNs 
    assert not data.isnull().values.any()
    
    F1_features = torch.tensor(data[F1_features_names].values)
    R1 = torch.tensor(data[R1_names].values)
    R2 = torch.tensor(data[R2_names].values)
    V = torch.tensor(data[V_names].values)
    C_ort = torch.tensor(data[C_ort_names].values)
    C = torch.tensor(data[C_names].values)

    F1_features = F1_features_scaling.normalize(F1_features)
    V = V_scaling.normalize(V)
    C_ort = C_ort_scaling.normalize(C_ort)
    C = C_scaling.normalize(C)
    
    dataset =  TensorDataset(F1_features.float(), R1.float(), V.float(), R2.float(), C_ort.float(), C.float())
    l1 = round(len(dataset)*traintest_split)
    l2 = len(dataset) - l1
    print('train/test: ',[l1,l2],'\n\n')
    train_set, test_set = torch.utils.data.random_split(dataset, [l1,l2], generator=torch.Generator().manual_seed(42))
    return train_set, test_set

def getDataset_pred(C_scaling,E,dataPath_pred):
    
    data = pd.read_csv(dataPath_pred)
    
    print('Data: ',data.shape)       
    # check for NaNs 
    assert not data.isnull().values.any()
    
    C = torch.tensor(data[C_names].values)
    # normalize stiffness by Young's modulus of base material
    C = torch.div(C,E)
    C = C_scaling.normalize(C)
    dataset =  C.float()
    return dataset