import torch

def computeR2(T1,T2):
    # T1: predictions
    # T2: ground truth
    cols = T1.shape[1]
    R2 = torch.zeros(cols)
    for i in range(cols):
        y = T1[:,i:(i+1)]
        x = T2[:,i:(i+1)]
        SSres = torch.norm(x-y)**2
        SStot = torch.norm(x-torch.mean(x))**2
        R2[i] = 1.-SSres/SStot
    return R2

def compute_NMSE(truth,pred):
    diff = truth-pred
    rel_error = compute_squared_error(diff)/compute_squared_error(truth)
    return rel_error

def compute_squared_error(matrix):
    norm = matrix[:,0]**2 + 2.*matrix[:,1]**2 + 2.*matrix[:,2]**2 + 2.*matrix[:,3]**2 + 2.*matrix[:,4]**2 + 2.*matrix[:,5]**2 + matrix[:,6]**2
    + 2.*matrix[:,7]**2 + 2.*matrix[:,8]**2 + 2.*matrix[:,9]**2 + 2.*matrix[:,10]**2 + matrix[:,11]**2 + 2.*matrix[:,12]**2 + 2.*matrix[:,13]**2
    + 2.*matrix[:,14]**2 + matrix[:,15]**2 + 2.*matrix[:,16]**2 + 2.*matrix[:,17]**2 + matrix[:,18]**2 + 2.*matrix[:,19]**2 + matrix[:,20]**2
    return norm