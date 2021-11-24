import os
import torch
from torch.utils.data import DataLoader
from parameters import *
from loadDataset import *
from normalization import decodeOneHot
from model import *
from voigt_rotation import *
from errorAnalysis import computeR2

if __name__ == '__main__':
    
    # torch.manual_seed(1234)
    # os.system('mkdir models')
    # os.system('mkdir TestReg')
    # os.system('mkdir TestReg/history')

    # Load and preprocess data
    F1_features_scaling, C_ort_scaling, C_scaling, V_scaling, C_hat_scaling = getSavedNormalization()
    train_set, test_set = getDataset(F1_features_scaling, V_scaling, C_ort_scaling, C_scaling)
    train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set))
    # Note: for test, batch_size=len(test_set) so that we load the entire test set at once
    F1_features_test, R1_test, V_test, R2_test, C_ort_test, C_test = next(iter(test_data_loader))
    print('\n-------------------------------------')
        
    # set softmax temperature (for stochastic inverse prediction)
    t = 1.

    ## load first forward model (F1)
    F1 = torch.load("models/F1.pt",map_location=device)
    F1.eval()

    ## load second forward model (F2)
    F2 = torch.load("models/F2.pt",map_location=device)
    F2.eval()

    ## load inverse model (G1 & G2)
    G1 = torch.load("models/G1.pt",map_location=device)
    G2 = torch.load("models/G2.pt",map_location=device)
    G1.eval(), G2.eval()

    F1_features_test, R1_test, V_test, R2_test, C_ort_test, C_test = next(iter(test_data_loader))
    with torch.no_grad():
        ## Testing
        # if required, raise temperature for larger variety of predictions
        # t = 2.
        F1_features_test, R1_test, R2_test, C_ort_test, C_test, V_test = F1_features_test.to(device), R1_test.to(device), R2_test.to(device), C_ort_test.to(device), C_test.to(device), V_test.to(device)
        rho_U_test_pred, V_test_pred, R1_test_pred, R2_test_pred, topology_test_pred = invModel_output(G1,G2,C_test,t,'gumbel')
        
        # forward prediction based on inverse designed lattice
        C_ort_test_pred_pred = F1(F1_features_test_pred)
        F2_features_test_pred_pred = assemble_F2_features(C_ort_test_pred_pred,R1_test_pred,V_test_pred,C_ort_scaling,method='6D')
        C_hat_test_pred_pred = F2(F2_features_test_pred_pred)
        C_test_pred_pred = rotate_C(C_hat_test_pred_pred, R2_test_pred, C_scaling, C_hat_scaling,method='6D')

        ###############---Export ---##############
        print('\nExporting:')

        ###############---Push tensors back to CPU ---##############

        # #decode one-hot-encoding
        rho_U_test_pred, topology_test_pred = torch.split(F1_features_test_pred, [4,27], dim=1)
        topology_pred = decodeOneHot(topology_test_pred,'shift')
        F1_features_test_pred = torch.cat((rho_U_test_pred,topology_pred),dim=1)

        R1_test_pred_angle_axis = rotation_6d_to_angleaxis(R1_test_pred)
        R2_test_pred_angle_axis = rotation_6d_to_angleaxis(R2_test_pred)
        
        # scale data to original range
        C_test = C_scaling.unnormalize(C_test)
        C_test_pred_pred = C_scaling.unnormalize(C_test_pred_pred)
        F1_features_test_pred = F1_features_scaling.unnormalize(F1_features_test_pred)
        V_test_pred = V_scaling.unnormalize(V_test_pred)

        full_pred = torch.cat((F1_features_test_pred,R1_test_pred_angle_axis,R2_test_pred_angle_axis,V_test_pred),dim=1)

        full_pred = full_pred.cpu()
        C_test = C_test.cpu()
        C_test_pred_pred = C_test_pred_pred.cpu()
        
        exportTensor("TestReg/full_pred",full_pred,all_names)
        exportTensor("TestReg/C_test",C_test,C_names)
        exportTensor("TestReg/C_test_pred_pred",C_test_pred_pred,C_names)
        print('--------------------------------------------\n')