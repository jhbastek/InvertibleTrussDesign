import torch
import torch.nn.functional as F
from train_parameters import *
from voigt_rotation import *
import pickle, io

# unpickle object also with a CPU-only machine, see issue: https://github.com/pytorch/pytorch/issues/16797
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def getActivation(activ):
    if(activ == 'relu'):
        sigma = torch.nn.ReLU()
    elif(activ == 'tanh'):
        sigma = torch.nn.Tanh()
    elif(activ == 'sigmoid'):
        sigma = torch.nn.Sigmoid()
    elif(activ == 'leaky'):
        sigma = torch.nn.LeakyReLU()
    elif(activ == 'softplus'):
        sigma = torch.nn.Softplus()
    elif(activ == 'logsigmoid'):
        sigma = torch.nn.LogSigmoid()
    elif(activ == 'elu'):
        sigma = torch.nn.ELU()
    elif(activ == 'gelu'):
        sigma = torch.nn.GELU()
    elif(activ == 'none'):
        sigma = torch.nn.Identity()
    else:
        raise ValueError('Incorrect activation function')
    return sigma

def createNN(inputDim,arch,outputDim,bias=True):
    model = torch.nn.Sequential()
    currDim = inputDim
    layerCount = 1
    activCount = 1
    for i in range(len(arch)):
        if(type(arch[i]) == int):
            model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,arch[i],bias=bias))
            currDim = arch[i]
            layerCount += 1
        elif(type(arch[i]) == str):
            model.add_module('activ '+str(activCount),getActivation(arch[i]))
            activCount += 1
    model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,outputDim,bias=bias))
    return model

def softmax(input, t):
    return F.log_softmax(input/t, dim=1)

def gumbel(input, t):
    return F.gumbel_softmax(input, tau=t, hard=True, eps=1e-10, dim=1)
    
def assemble_F2_features(C_ort,R1,V,C_ort_scaling,method=None):
    # scale C_ort to its original range
    C_ort_unscaled = C_ort_scaling.unnormalize(C_ort)
    # rotate C_ort (directly in Voigt notation)
    C_tilde = direct_rotate(C_ort_unscaled,R1,orthotropic=True,method=method)
    return torch.cat((C_tilde,V),dim=1)

def invModel_output(G1,G2,input,t,activation):
    # continuous params: [stretch1, stretch2, stretch3, rot_stretch1, rot_stretch2, rot_stretch3, theta, rot_ax1, rot_ax2]
    topology1,topology2,topology3,rep1,rep2,rep3 = torch.split(G1(input), [7,7,7,2,2,2], dim=1)
    m = getActivation('sigmoid')
    if(activation == 'one-hot'):
        # enforce one-hot encoding by small temperature
        t = 1.e-6
    if(activation == 'softmax' or activation == 'one-hot'):
        topology = torch.cat((softmax(topology1,t),softmax(topology2,t),softmax(topology3,t),softmax(rep1,t),softmax(rep2,t),softmax(rep3,t)), dim=1)
    elif(activation == 'gumbel'):
        topology1,topology2,topology3,rep1,rep2,rep3 = softmax(topology1,t),softmax(topology2,t),softmax(topology3,t),softmax(rep1,t),softmax(rep2,t),softmax(rep3,t)
        topology = torch.cat((gumbel(topology1,t),gumbel(topology2,t),gumbel(topology3,t),gumbel(rep1,t),gumbel(rep2,t),gumbel(rep3,t)), dim=1)
    else:
        raise ValueError('Incorrect activation function')

    features = torch.cat((topology, input), dim=1)
    rho_U, V, rot1, rot2 = torch.split(G2(features), [4,3,6,6], dim=1)
    # scale to [0,1] using sigmoid
    rho_U, V = m(rho_U), m(V)

    return rho_U, V, rot1, rot2, topology
    
def rotate_C(C_in,R,C_in_scaling,C_out_scaling,method=None):
    temp = C_in_scaling.unnormalize(C_in)
    temp = direct_rotate(temp,R,method=method)
    C = C_out_scaling.normalize(temp)
    return C