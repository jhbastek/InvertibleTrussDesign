import pathlib
import torch
from torch.utils.data import DataLoader
from parameters import *
from loadDataset import *
from normalization import decodeOneHot
from model_utils import *
from voigt_rotation import *
from errorAnalysis import computeR2

if __name__ == '__main__':
    
    torch.manual_seed(1234)
    
    # create directories
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('Training').mkdir(exist_ok=True)
    pathlib.Path('Training/history').mkdir(exist_ok=True)

    # Load and preprocess data
    F1_features_scaling, C_ort_scaling, C_scaling, V_scaling, C_hat_scaling = getNormalization()
    train_set, test_set = getDataset(F1_features_scaling, V_scaling, C_ort_scaling, C_scaling)
    train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set))
    # Note: for test, batch_size=len(test_set) so that we load the entire test set at once
    F1_features_test, R1_test, V_test, R2_test, C_ort_test, C_test = next(iter(test_data_loader))
    print('\n-------------------------------------')
        
    # set softmax temperature (for stochastic inverse prediction)
    t = 1.

    ## first forward model (F1)
    if(F1_train):
        # initialize first forward model
        F1 = createNN(31,F1_arch,9).to(device)
        # set up optimizer
        F1_optimizer = torch.optim.Adam(F1.parameters(), lr=F1_learning_rate)
        F1_train_history, F1_test_history = [],[]
        # training
        for F1_epoch_iter in range(F1_train_epochs):
            F1_train_loss = 0.
            for iteration, batch in enumerate(train_data_loader,0):
                # get batch
                F1_features_train, C_ort_train = batch[0].to(device), batch[4].to(device)
                # set train mode
                F1.train()
                # forward pass F1
                C_ort_train_pred = F1(F1_features_train)
                # compute loss
                fwdLoss = lossFn(C_ort_train_pred,C_ort_train)
                # optimize
                F1_optimizer.zero_grad()
                fwdLoss.backward()
                F1_optimizer.step()
                # store (batch) training loss
                F1_train_loss = fwdLoss.item()
            F1_features_test, C_ort_test = F1_features_test.to(device), C_ort_test.to(device)
            C_ort_test_pred = F1(F1_features_test)
            F1_test_loss = lossFn(C_ort_test_pred,C_ort_test).item()
            print("| {}:{}/{} | F1_EpochTrainLoss: {:.2e} | F1_EpochTestLoss: {:.2e}".format("F1",F1_epoch_iter,F1_train_epochs,F1_train_loss,F1_test_loss))
            F1_train_history.append(F1_train_loss)
            F1_test_history.append(F1_test_loss)
        print('\n-------------------------------------')
        # save model
        torch.save(F1,"models/F1.pt")
        # export loss history
        exportList('Training/history/F1_train_history',F1_train_history)
        exportList('Training/history/F1_test_history',F1_test_history)
    else:
        F1 = torch.load("models/F1.pt",map_location=device)
    F1.eval()

    ## second forward model (F2)
    if(F2_train):
        # initialize second forward model
        F2 = createNN(24, F2_arch, 21).to(device)
        # set up optimizer
        F2_optimizer = torch.optim.Adam(F2.parameters(), lr=F2_learning_rate)
        F2_train_history,F2_test_history = [],[]
        # training
        for F1_epoch_iter in range(F2_train_epochs):
            F2_train_loss = 0.
            for iteration, batch in enumerate(train_data_loader,0):
                # get batch
                F1_features_train,R1_train,V_train,R2_train,C_ort_train,C_train = \
                    batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device),batch[4].to(device),batch[5].to(device)
                # set train mode
                F2.train()
                # forward pass F1
                C_ort_train_pred = F1(F1_features_train)
                # construct input for F2
                F2_features_train_pred = assemble_F2_features(C_ort_train_pred,R1_train,V_train,C_ort_scaling)
                # forward pass F2
                C_hat_train_pred = F2(F2_features_train_pred)
                # rotate with given R2 to obtain C
                C_train_pred = rotate_C(C_hat_train_pred,R2_train,C_hat_scaling,C_scaling)
                # compute loss
                fwdLoss = lossFn(C_train_pred,C_train)
                # optimize
                F2_optimizer.zero_grad()
                fwdLoss.backward()
                F2_optimizer.step()
                # store (batch) training loss
                F2_train_loss = fwdLoss.item()
            F1_features_test, R1_test, V_test, R2_test, C_ort_test, C_test = \
                F1_features_test.to(device), R1_test.to(device), V_test.to(device), R2_test.to(device), C_ort_test.to(device), C_test.to(device)
            # same as above but using the test data
            C_ort_test_pred = F1(F1_features_test)
            F2_features_test_pred = assemble_F2_features(C_ort_test_pred,R1_test,V_test,C_ort_scaling)
            C_hat_test_pred = F2(F2_features_test_pred)
            C_test_pred = rotate_C(C_hat_test_pred,R2_test,C_hat_scaling,C_scaling)
            F2_test_loss = lossFn(C_test_pred,C_test).item()
            print("| {}:{}/{} | F2_EpochTrainLoss: {:.2e} | F2_EpochTestLoss: {:.2e}".format("F2",F1_epoch_iter,F2_train_epochs, F2_train_loss, F2_test_loss))
            F2_train_history.append(F2_train_loss)
            F2_test_history.append(F2_test_loss)
        print('\n-------------------------------------')
        # save model
        torch.save(F2, "models/F2.pt")
        # export loss history
        exportList('Training/history/F2_train_history',F2_train_history)
        exportList('Training/history/F2_test_history',F2_test_history)
    else:
        F2 = torch.load("models/F2.pt",map_location=device)
    F2.eval()

    ## inverse model (G1 & G2)
    if(inv_train):
        # initialize inverse model
        G1 = createNN(21,inv_arch,27).to(device) # 21 topology parameters, 1 relative density, 6 stretches, 2 x 6 rotation paramters (6D representation)
        G2 = createNN(21+27,inv_arch,19).to(device)
        # set up optimizer
        inv_optimizer = torch.optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=inv_learning_rate)
        inv_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inv_optimizer, 'min', patience=20, factor=0.5)
        inv_train_history,inv_test_history = [],[]
        # training
        for inv_epoch_iter in range(inv_train_epochs):
            inv_train_loss = 0.        
            for iteration, batch in enumerate(train_data_loader,0):
                # get batch
                F1_features_train,R1_train,V_train,R2_train,C_ort_train,C_train = \
                    batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device),batch[4].to(device),batch[5].to(device)
                # set train mode
                G1.train(),G2.train()
                # predict
                rho_U_train_pred, V_train_pred, R1_train_pred, R2_train_pred, topology_train_pred = invModel_output(G1,G2,C_train,t,'gumbel')
                # construct input for F1
                F1_features_train_pred = torch.cat((rho_U_train_pred,topology_train_pred),dim=1)
                # forward pass F1
                C_ort_train_pred = F1(F1_features_train_pred)
                # construct input for F2
                F2_features_train_pred = assemble_F2_features(C_ort_train_pred,R1_train_pred,V_train_pred,C_ort_scaling,method='6D')
                # forward pass F2
                C_hat_train_pred_pred = F2(F2_features_train_pred)
                # apply second rotation (using 6D representation)
                C_train_pred_pred = rotate_C(C_hat_train_pred_pred,R2_train_pred,C_hat_scaling,C_scaling,method='6D')
                # compute loss
                invLoss = lossFn(C_train_pred_pred, C_train)
                # optimize
                inv_optimizer.zero_grad()
                invLoss.backward()
                inv_optimizer.step()
                # store (batch) training loss
                inv_train_loss = invLoss.item()
            C_test = C_test.to(device)
            # same as above but using the test data
            rho_U_test_pred, V_test_pred, R1_test_pred, R2_test_pred, topology_test_pred = invModel_output(G1,G2,C_test,t,'gumbel')
            F1_features_test_pred = torch.cat((rho_U_test_pred, topology_test_pred), dim=1)
            C_ort_test_pred = F1(F1_features_test_pred)
            F2_features_test_pred = assemble_F2_features(C_ort_test_pred,R1_test_pred,V_test_pred,C_ort_scaling,method='6D')
            C_hat_test_pred_pred = F2(F2_features_test_pred)
            C_test_pred_pred = rotate_C(C_hat_test_pred_pred, R2_test_pred, C_hat_scaling, C_scaling,method='6D')
            invTestLoss = lossFn(C_test_pred_pred,C_test).item()
            inv_scheduler.step(invTestLoss)
            print("| {}:{}/{} | lr: {:.2e} | invEpochTrainLoss: {:.2e} | invEpochTestLoss: {:.2e}".format("inv",inv_epoch_iter, inv_train_epochs, inv_optimizer.param_groups[0]['lr'], inv_train_loss, invTestLoss))
            inv_train_history.append(inv_train_loss)
            inv_test_history.append(invTestLoss)
        print('\n-------------------------------------')
        # save models
        torch.save(G1,"models/G1.pt"), torch.save(G2,"models/G2.pt")
        # export loss histories
        exportList('Training/history/inv_train_history',inv_train_history)
        exportList('Training/history/inv_test_history',inv_test_history)
    else:
        G1 = torch.load("models/G1.pt",map_location=device)
        G2 = torch.load("models/G2.pt",map_location=device)
    G1.eval(), G2.eval()

    ## testing
    with torch.no_grad():
        F1_features_test, R1_test, R2_test, C_ort_test, C_test, V_test = F1_features_test.to(device), R1_test.to(device), R2_test.to(device), C_ort_test.to(device), C_test.to(device), V_test.to(device)
        # inverse prediction
        rho_U_test_pred, V_test_pred, R1_test_pred, R2_test_pred, topology_test_pred = invModel_output(G1,G2,C_test,t,'gumbel')
        # assemble F1 features based on output of inverse model
        F1_features_test_pred = torch.cat((rho_U_test_pred, topology_test_pred), dim=1)

        # forward prediction based on given lattice
        C_ort_test_pred = F1(F1_features_test)
        F2_features_test_pred = assemble_F2_features(C_ort_test_pred,R1_test,V_test,C_ort_scaling)
        C_hat_test_pred = F2(F2_features_test_pred)
        C_test_pred = rotate_C(C_hat_test_pred, R2_test, C_hat_scaling, C_scaling)
        
        # forward prediction based on inversely designed lattice
        C_ort_test_pred_pred = F1(F1_features_test_pred)
        F2_features_test_pred_pred = assemble_F2_features(C_ort_test_pred_pred,R1_test_pred,V_test_pred,C_ort_scaling,method='6D')
        C_hat_test_pred_pred = F2(F2_features_test_pred_pred)
        C_test_pred_pred = rotate_C(C_hat_test_pred_pred, R2_test_pred, C_hat_scaling, C_scaling,method='6D')

        # compute R2 values
        print('\nR2 values:\n--------------------------------------------')
        F1ComponentR2 = computeR2(C_ort_test_pred, C_ort_test)
        print('F1 test C R2:',F1ComponentR2,'\n')
        F2ComponentR2Y = computeR2(C_test_pred, C_test)
        print('F2 test C R2:',F2ComponentR2Y,'\n')
        invComponentR2Y = computeR2(C_test_pred_pred, C_test)
        print('Inverse test reconstruction C R2:',invComponentR2Y,'\n')

        ## export for post-processing
        print('\nExporting:')

        # #decode one-hot-encoding into original lattice nomenclature (test set)
        rho_U_test, topology_test = torch.split(F1_features_test, [4,27], dim=1)
        topology = decodeOneHot(topology_test)
        F1_features_test = torch.cat((rho_U_test,topology),dim=1)

        # #decode one-hot-encoding into original lattice nomenclature (inverse prediction)
        rho_U_test_pred, topology_test_pred = torch.split(F1_features_test_pred, [4,27], dim=1)
        topology_pred = decodeOneHot(topology_test_pred)
        F1_features_test_pred = torch.cat((rho_U_test_pred,topology_pred),dim=1)

        # decode 6D representation into angle-axis representation
        R1_test_pred_angle_axis = rot6DToAngleAxis(R1_test_pred)
        R2_test_pred_angle_axis = rot6DToAngleAxis(R2_test_pred)
        
        # scale data to original range
        C_test = C_scaling.unnormalize(C_test)
        C_test_pred = C_scaling.unnormalize(C_test_pred)
        C_test_pred_pred = C_scaling.unnormalize(C_test_pred_pred)
        F1_features_test_pred = F1_features_scaling.unnormalize(F1_features_test_pred)
        V_test_pred = V_scaling.unnormalize(V_test_pred)

        # construct full descriptor of inversely designed lattice
        full_pred = torch.cat((F1_features_test_pred,R1_test_pred_angle_axis,R2_test_pred_angle_axis,V_test_pred),dim=1)

        # push tensors back to cpu
        full_pred = full_pred.cpu()
        C_test = C_test.cpu()
        C_test_pred = C_test_pred.cpu()
        C_test_pred_pred = C_test_pred_pred.cpu()
        
        # export tensors to .csv
        exportTensor("Training/full_pred",full_pred,all_names)
        exportTensor("Training/C_test",C_test,C_names)
        exportTensor("Training/C_test_pred",C_test_pred,C_names)
        exportTensor("Training/C_test_pred_pred",C_test_pred_pred,C_names)
        print('Finished.\n')