
import torch
import numpy as np

# sigma.shape = (raysxSamples) ; c.shape (rays, Samples, color) ; t.shape = (rays, Nc)  
def c_pred(sigma, c, t):
    
    delta = torch.diff(t).to(device=c.device)    
    
    Ti = torch.exp( - torch.cumsum( sigma[:, :-1] * delta, dim=-1)).to(device=c.device) 
    
    wi = Ti[..., None] * (1 - torch.exp(-sigma[:, :-1, None] * delta[..., None]))
    
    c_hat = torch.sum( wi * c[:, :-1 , :], dim=1)

    return c_hat, wi

'''
    sigma (rays, samples)
    c (rays, samples, rgb)
    t = (rays, samples)
    
    # rays, samples-1
    delta = torch.diff(t)
    
    # (rays, samples-1)
    Ti = torch.exp( - torch.cumsum( sigma[:, :-1] * delta, dim=-1)) # rays, samples-1
    alpha_i = (1 - torch.exp( - sigma[:, :-1] * delta)) # rays, samples-1 
    wi = Ti * alpha_i # rays, sample-1
    
    c_hat = torch.sum( wi[..., None] * c[:, :-1 , :], dim=1) 

'''


def stratified_sampling_rays(N, tn, tf, rays=3):
    '''
    Stratified samping for a set of rays, returns (rays, N)
    '''
    samples = np.random.uniform(low=0, high=(1/N)*(tf - tn), size=(rays, N))
    i = np.expand_dims(np.arange(0, N), 0)
    t = torch.tensor(tn + (i/N)*(tf - tn) + samples, dtype=torch.float32, device="cuda")
    
    return t
