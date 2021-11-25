import torch
from torch.utils.data import DataLoader
from parameters import *
from loadDataset import *
from normalization import decodeOneHot
from model_utils import *
from voigt_rotation import *
from errorAnalysis import compute_NMSE

if __name__ == '__main__':
    
    torch.manual_seed(1234)

    ## load and preprocess data
    # load normalization (based on training dataset)
    F1_features_scaling, C_ort_scaling, C_scaling, V_scaling, C_hat_scaling = getSavedNormalization()
    pred_set = getDataset_pred(C_scaling)
    pred_set_loader = DataLoader(dataset=pred_set, num_workers=numWorkers, batch_size=len(pred_set))
    # Note: for test, batch_size=len(test_set) so that we load the entire test set at once
    C_target = next(iter(pred_set_loader))
    print('\n-------------------------------------')
        
    # set softmax temperature (for stochastic inverse prediction)
    t = 100.

    ## load models
    # load F1
    F1 = torch.load("models/F1.pt",map_location=device)
    F1.eval()
    # load F2
    F2 = torch.load("models/F2.pt",map_location=device)
    F2.eval()
    # load inverse model (G1 & G2)
    G1 = torch.load("models/G1.pt",map_location=device)
    G2 = torch.load("models/G2.pt",map_location=device)
    G1.eval(), G2.eval()

    repetitions = 200
    num_samples = len(C_target)
    top_lat = 5

    # initialize nested lists to collect best predictions
    top_C_target_pred_pred = [[] for i in range(num_samples)]
    top_full_target_pred = [[] for i in range(num_samples)]

    with torch.no_grad():
        C_target = C_target.to(device)

        # repeat target lables to generate large variety of predictions
        C_target = C_target.repeat(1,repetitions).view(-1,21)
        # inverse prediction
        rho_U_target_pred, V_target_pred, R1_target_pred, R2_target_pred, topology_target_pred = invModel_output(G1,G2,C_target,t,'gumbel')
        # assemble F1 features based on output of inverse model
        F1_features_target_pred = torch.cat((rho_U_target_pred, topology_target_pred), dim=1)
        # forward prediction based on inversely designed lattice
        C_ort_target_pred_pred = F1(F1_features_target_pred)
        F2_features_target_pred_pred = assemble_F2_features(C_ort_target_pred_pred,R1_target_pred,V_target_pred,C_ort_scaling,method='6D')
        C_hat_target_pred_pred = F2(F2_features_target_pred_pred)
        C_target_pred_pred = rotate_C(C_hat_target_pred_pred, R2_target_pred, C_hat_scaling, C_scaling,method='6D')

        # assemble full descriptor
        full_target_pred = torch.cat((F1_features_target_pred,R1_target_pred,R2_target_pred,V_target_pred),dim=1)

        # scale stiffness to original range and compute NMSE
        C_target = C_scaling.unnormalize(C_target)
        C_target_pred_pred = C_scaling.unnormalize(C_target_pred_pred)
        rel_error = compute_NMSE(C_target,C_target_pred_pred)

        lowest_error = torch.zeros(num_samples,device=device)+1.e20
        for j in range(num_samples):
            for i in range(repetitions):
                cur_iter = j*repetitions + i
                if (rel_error[cur_iter] < lowest_error[j]):
                    print('Identified lattice with lower NMSE.')
                    top_C_target_pred_pred[j].append(C_target_pred_pred[cur_iter])
                    top_full_target_pred[j].append(full_target_pred[cur_iter])
                    lowest_error[j] = rel_error[cur_iter]

        selected_C_target_pred_pred = torch.zeros((num_samples,top_lat,21+1),device=device)
        for i, list in enumerate(top_C_target_pred_pred):
            temp = torch.stack(list[:-top_lat-1:-1])
            num_predictions = temp.shape[0]
            temp = torch.cat((torch.zeros(num_predictions,1)+i+1,temp),dim=1)
            selected_C_target_pred_pred[i,0:num_predictions] = temp

        selected_full_target_pred = torch.zeros((num_samples,top_lat,46+1),device=device)
        for i, list in enumerate(top_full_target_pred):
            temp = torch.stack(list[:-top_lat-1:-1])
            num_predictions = temp.shape[0]
            temp = torch.cat((torch.zeros(num_predictions,1)+i+1,temp),dim=1)
            selected_full_target_pred[i,0:temp.shape[0]] = temp

        selected_C_target_pred_pred = torch.flatten(selected_C_target_pred_pred,end_dim=1)
        selected_full_target_pred = torch.flatten(selected_full_target_pred,end_dim=1)

        # delete zero-rows
        selected_C_target_pred_pred = selected_C_target_pred_pred[selected_C_target_pred_pred[:,0]!=0]
        selected_full_target_pred = selected_full_target_pred[selected_full_target_pred[:,0]!=0]

        ## export for post-processing
        print('\nExporting:')

        # split prediction into subparts to rescale them to original range
        sample, rho_U_target_pred, topology_target_pred, R1_target_pred, R2_target_pred, V_target_pred = torch.split(selected_full_target_pred, [1,4,27,6,6,3], dim=1)

        # decode one-hot-encoding into original lattice nomenclature
        topology_target_pred = decodeOneHot(topology_target_pred)
        F1_features_target_pred = torch.cat((rho_U_target_pred,topology_target_pred),dim=1)

        # decode 6D representation into angle-axis representation
        R1_target_pred_angle_axis = rot6DToAngleAxis(R1_target_pred)
        R2_target_pred_angle_axis = rot6DToAngleAxis(R2_target_pred)
        
        # scale data to original range
        F1_features_target_pred = F1_features_scaling.unnormalize(F1_features_target_pred)
        V_target_pred = V_scaling.unnormalize(V_target_pred)

        # construct full descriptor of inversely designed lattice
        full_pred = torch.cat((sample,F1_features_target_pred,R1_target_pred_angle_axis,R2_target_pred_angle_axis,V_target_pred),dim=1)

        # push tensors back to cpu
        full_pred = full_pred.cpu()
        C_target = C_target.cpu()
        selected_C_target_pred_pred = selected_C_target_pred_pred.cpu()
        
        # export tensors for post-processing
        exportTensor("Predictions/full_pred",full_pred,['sample']+all_names)
        exportTensor("Predictions/C_target",C_target,C_names)
        exportTensor("Predictions/C_target_pred_pred",selected_C_target_pred_pred,['sample']+C_names)
        print('Finished.\n')