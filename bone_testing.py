# -*- coding: utf-8 -*-

if __name__ == '__main__':
    
    import os
    import torch
    from torch.utils.data import Dataset, TensorDataset, DataLoader
    import numpy as np
    import pandas as pd
    from parameters import *
    from normalization import Normalization
    from normalization import decodeOneHot
    from loadDataset import *
    from model import *
    from correction import *
    from errorAnalysis import *
    from voigt_rotation import *

    # torch.manual_seed(0)
    # os.system('mkdir models')
    # os.system('mkdir TestReg')
    # os.system('mkdir TestReg/history')

    ##############---FWD MODEL---##############
    # fwdModel = createNN(33, fwdArch, ort_labelDim) # new: 7+7+7+2+3+3 categorical features (one-hot) + 4 continuous features (stretches & relative density)
    fwdModel = torch.load("models/fwdModel.pt",map_location=device)
    fwdOptimizer = torch.optim.Adam(fwdModel.parameters(), lr=fwdLearningRate_1)    
    # print('\n\n**************************************************************')
    # print('fwdModel', fwdModel)
    # print('**************************************************************\n')

    ##############---SHEAR MODEL---##############
    # shearModel = createNN(28, shearArch, 21)
    shearModel = torch.load("models/shearModel.pt",map_location=device)
    # print('\n\n**************************************************************')
    # print('shearModel', shearModel)
    # print('**************************************************************\n')

    # #############---INVERSE MODEL---##############
    # t = torch.tensor(1., requires_grad=True, device=device)
    gaussian_features = 0
    # invModel = createNN(21, invArch, 15) # output: 9 ort. stiffness, 3 rot, 3 shear stretches
    # invModel = createNN(21+gaussian_features, invArch, 42) # 21 topology parameters, 1 relative density, 6 stretches, 6 rotation paramters (6D representation)
    invModel = torch.load("models/invModel.pt",map_location=device)
    invModel2 = torch.load("models/invModel2.pt",map_location=device)
    # print('\n\n**************************************************************')
    # print('invModel', invModel)
    # print('**************************************************************\n')

    ##############---INIT DATA---##############
    # train_set, test_set, featureNormalization, ort_labelNormalization, labelNormalization, labelInverseNormalization, shearNormalization, topologyNormalization = getDataset()    
    # train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize, shuffle=batchShuffle)
    # test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)

    # only considers subset of full data:
    data_samples_reduced = 3000

    # featureNormalization, ort_labelNormalization, labelNormalization, labelInverseNormalization, shearNormalization, topologyNormalization, unrotatedlabelNormalization = getNormalization()
    featureNormalization, ort_labelNormalization, labelNormalization, labelInverseNormalization, shearNormalization, topologyNormalization, unrotatedlabelNormalization = getSavedNormalization()
    test_set = getDataset_bones(data_samples_reduced, labelNormalization)
    # print(test_set)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)
    #Note: for test, batch_size=len(test_set) so that we load the entire test set at once

    ##############---Training---##############
    fwdEpochLoss = 0.0
    shearEpochLoss = 0.0

    fwdTrainHistory = []
    fwdTestHistory = []
    shearTrainHistory = []
    shearTestHistory = []
    invTrainHistory = []
    invTestHistory = []
    loader_all_test = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)
    y_all_test = next(iter(loader_all_test))
    print('\n-------------------------------------')

    fwdModel.to(device)
    shearModel.to(device)
    invModel.to(device)
    invModel2.to(device)

    fwdModel.eval()
    shearModel.eval()
    invModel.eval()
    invModel2.eval()

    #############---PRINT OUT TRAING AND TEST LOSS FOR shear MODEL---##############

    #############---TESTING---##############
    # load the entire test data from test_data_loader
    y_test = next(iter(test_data_loader))

    with torch.no_grad(): #Use no_grad only inferring/prediction and NOT training

        t = 2.
        # t = np.exp(-0.05*(110))

        y_test = y_test.to(device)

        # y_test = torch.div(y_test,0.01)
        continuous_test_pred, topology_test_pred = lastInverseShearLayerSplit_6D_density_new_2models(invModel,y_test,t,'gumbel',invModel2)
        # topology_test_pred = torch.exp(topology_test_pred)
        x_test_pred, shear_test_pred, rot_test_pred, rot_test_pred2 = torch.split(continuous_test_pred, [4,3,6,6], dim=1)
        x_test_pred = torch.cat((x_test_pred, topology_test_pred), dim=1)

        continuous_test_pred_hard, topology_test_pred_hard = lastInverseShearLayerSplit_6D_density_new_2models(invModel,y_test,t,'one-hot',invModel2)
        topology_test_pred_hard = torch.exp(topology_test_pred_hard)
        x_test_pred_hard, shear_test_pred_hard, rot_test_pred_hard, rot_test_pred_hard2 = torch.split(continuous_test_pred_hard, [4,3,6,6], dim=1)
        x_test_pred_hard = torch.cat((x_test_pred_hard, topology_test_pred_hard), dim=1)

        # Fix or post-process any predictions; only during post-process and shouldn't be done during training

        # x_test_pred = correctshearPredictions(x_test_pred) # This is the final prediction, not the uncorrected one
        
        pred_shear_features = assembleReducedShearFeatures_6D(fwdModel,x_test_pred,rot_test_pred,shear_test_pred,ort_labelNormalization)
        y_test_pred_pred = shearModel(pred_shear_features)
        y_test_pred_pred_rot = rotate_labels_6D(y_test_pred_pred, rot_test_pred2, labelNormalization, unrotatedlabelNormalization)

        pred_shear_features_hard = assembleReducedShearFeatures_6D(fwdModel,x_test_pred_hard,rot_test_pred_hard,shear_test_pred_hard,ort_labelNormalization)
        y_test_pred_pred_hard = shearModel(pred_shear_features_hard)
        y_test_pred_pred_hard_rot = rotate_labels_6D(y_test_pred_pred_hard, rot_test_pred_hard2, labelNormalization, unrotatedlabelNormalization)

        #############---POST PROC---##############
        # print('\n\nERRORS:\n--------------------------------------------')
        # fwdError = computeError(y_test_pred, y_test) 
        # fwdComponentErrors = computeComponentErrors(y_test_pred,y_test)
        # print('Fwd test Y err: {:.6e}'.format(fwdError))
        # print('Component-wise:',fwdComponentErrors)
        # print('\n')
        
        # shearErrorY = computeError(y_test_pred, y_test)
        # shearComponentErrorsY = computeComponentErrors(y_test_pred,y_test)
        # print('shear test reconstruction Y err: {:.6e}'.format(shearErrorY))
        # print('Component-wise:',shearComponentErrorsY)
        # print('\nshear test Y err using FEM: please run simulations')
        # print('\n')
        
        # shearErrorX = computeError(x_test_pred, x_test)
        # shearComponentErrorsX = computeComponentErrors(x_test_pred,x_test)
        # print('shear test prediction X err: {:.6e}'.format(shearErrorX))
        # print('Component-wise:', shearComponentErrorsX)
        # print('^^ Dont freak out; this is expected to be (very) large')
        # print('--------------------------------------------\n')

        print('\nPredictions:\n--------------------------------------------')

        y_test = labelNormalization.unnormalize(y_test)
        y_test_pred_pred_rot = labelNormalization.unnormalize(y_test_pred_pred_rot)
        y_test_pred_pred_hard_rot = labelNormalization.unnormalize(y_test_pred_pred_hard_rot)

        # print(y_test)
        # print(y_test_pred_pred_rot)

        # invComponentR2Y = computeR2(y_test_pred_pred, y_test)
        # print('Inv test reconstruction Y R2:',invComponentR2Y,'\n')

        # print('shear test reconstruction Y R2 using FEM: please run simulations\n')

        # shearComponentR2X = computeR2(x_test_pred, x_test)
        # print('shear test prediction X R2:',shearComponentR2X)
        # print('^^ Dont freak out; this is expected to be (very) low')
        # print('--------------------------------------------\n')
        
        ###############---UNNORMALIZE Features for exporting (only) to non-ML world---##############
        # x_test = featureNormalization.unnormalize(x_test)
        # x_test_pred_uncorrected = featureNormalization.unnormalize(x_test_pred_uncorrected)
        # x_test_pred = featureNormalization.unnormalize(x_test_pred)

        ###############---Export ---##############
        print('\nExporting:')

        ###############---Push tensors back to CPU ---##############

        # #decode one-hot-encoding
        cont_pred, topology_test_pred = torch.split(x_test_pred, [4,27], dim=1)
        topology_pred = decodeOneHot(topology_test_pred,'shift')
        x_test_pred = torch.cat((cont_pred,topology_pred),dim=1)
        
        cont_pred_hard, topology_test_pred_hard = torch.split(x_test_pred_hard, [4,27], dim=1)
        topology_pred_hard = decodeOneHot(topology_test_pred_hard,'shift')
        x_test_pred_hard = torch.cat((cont_pred_hard,topology_pred_hard),dim=1)

        angle_axis_pred = rotation_6d_to_angleaxis(rot_test_pred)
        angle_axis_pred2 = rotation_6d_to_angleaxis(rot_test_pred2)

        # x_test = featureNormalization.unnormalize(x_test)
        x_test_pred = featureNormalization.unnormalize(x_test_pred)
        x_test_pred_hard = featureNormalization.unnormalize(x_test_pred_hard)
        shear_test_pred = shearNormalization.unnormalize(shear_test_pred)

        full_pred = torch.cat((x_test_pred,angle_axis_pred,angle_axis_pred2,shear_test_pred),dim=1)

        # x_test_pred = x_test_pred.cpu()
        # shear_test_pred = shear_test_pred.cpu()
        # angle_axis_pred = angle_axis_pred.cpu() 
        full_pred = full_pred.cpu()
        x_test_pred_hard = x_test_pred_hard.cpu()
        y_test = y_test.cpu()
        # y_test_pred= y_test_pred.cpu()
        y_test_pred_pred_rot = y_test_pred_pred_rot.cpu()
        y_test_pred_pred_hard_rot = y_test_pred_pred_hard_rot.cpu()
        
        # exportTensor("TestReg/bone/x_test_pred",x_test_pred,featureNames)
        # exportTensor("TestReg/bone/shear_test_pred",shear_test_pred,shearNames)
        # exportTensor("TestReg/bone/angle_axis_pred",angle_axis_pred,rotationNames)
        exportTensor_rem_dupl("TestReg/bone/full_pred",full_pred,fullNames)
        # exportTensor("TestReg/bone/x_test_pred_hard",x_test_pred_hard,featureNames)

        # exportTensor("TestReg/bone/y_test",y_test,labelNames)
        # exportTensor("TestReg/bone/y_test_pred_pred_rot",y_test_pred_pred_rot,labelNames)
        # exportTensor("TestReg/bone/y_test_pred_pred_hard_rot",y_test_pred_pred_hard_rot,labelNames)

        # exportTensor("TestReg/rot_test",rot_test,rotationNames)
        # # exportTensor("TestReg/y_rot_test",y_rot_test,labelrotationNames)
        # # exportTensor("TestReg/y_rot_test_pred",y_rot_test_pred,labelrotationNames)
        # print('--------------------------------------------\n')
        # print('--------------------------------------------\n')


