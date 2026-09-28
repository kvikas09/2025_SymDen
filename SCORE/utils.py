import os
import math
import time
import torch
import pickle
import numpy as np

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd

import torch.nn.functional as functional

torch_device = 'cuda'

def log_prob(x,m2 = -4. ,lamda = 4.0):
    x.requires_grad_(True)
    potential = m2*x**2 + lamda*x**4
    Nd = len(x.shape)-1
    dims = range(2,Nd+1)
    for mu in dims:
        potential += 2*x**2
        potential -= x*torch.roll(x,-1,mu)
        potential -= x*torch.roll(x,1,mu)
    return -torch.sum(potential , dim = [1,2,3]).to(device = x.device)

def force(x):
    x = x.requires_grad_(True)
    e = log_prob(x)
    return torch.autograd.grad(e.sum(), x)[0]


def jacobian(f, x):
    """
    Computes the Jacobian
    """
    
    y = f(x)
    v = torch.zeros_like(y,device = torch_device)
    v[:] = 1.
    dy_dx = autograd.grad(y, x, grad_outputs=v, retain_graph=True,
                                create_graph=True, allow_unused=True)[0]  # shape [B, N]
    jacobian = dy_dx.requires_grad_().to(torch_device)
    return jacobian

class SimpleNormal:
    def __init__(self, loc, var, device = torch_device):
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
        self.device = device
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1)).to(self.device)
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,)).to(self.device)
        return x.reshape(batch_size, *self.shape)


class Affine_Coupling(nn.Module):
    def __init__(self,input_shape, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Conv2d(self.input_shape[0], self.hidden_dim, kernel_size = 3, stride=1)
        self.layer2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size = 3, stride=1)
        self.scale  = nn.Conv2d(self.hidden_dim, self.input_shape[0],kernel_size = 3, stride=1)
        self.translation = nn.Conv2d(self.hidden_dim, self.input_shape[0],kernel_size = 3, stride=1)
        self.circular_pad = nn.CircularPad2d(1)

    def _compute(self, x):
        out = torch.relu(self.layer1(self.circular_pad(x)))
        out = torch.relu(self.layer2(self.circular_pad(out)))
        return out
    
    def forward(self, x):
        ## convert latent space variable to observed variable
        out = self._compute(x)
        s   = torch.tanh(self.scale(self.circular_pad(out)))
        t   = self.translation(self.circular_pad(out))
        
        return s,t

def checkerboard(height, width, reverse=False, dtype=torch.float32):
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)] 
    checkerboard = torch.tensor(checkerboard, dtype = dtype)
    if reverse:
        checkerboard = 1 - checkerboard
    
    checkerboard = torch.reshape(checkerboard, (1,1,height,width))
        
    return checkerboard


class RNVP(nn.Module):
    '''
    A RealNVP class for modeling 2 dimensional distributions
    '''
    def __init__(self, num_coupling_layers,input_shape,hidden_dim, var):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(RNVP, self).__init__()   
        self.var = var
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim  
        self.num_coupling_layers = num_coupling_layers
        self.distribution = SimpleNormal(torch.zeros((self.input_shape),device = torch_device), self.var*torch.ones((self.input_shape),device = torch_device), device = torch_device)
        self.layers_list = nn.ModuleList([Affine_Coupling(input_shape,hidden_dim) for i in range(num_coupling_layers)])
        self.masks = torch.tensor(np.array([checkerboard(input_shape[1],input_shape[2], reverse=False),checkerboard(input_shape[1],input_shape[2], reverse=True)]*(num_coupling_layers // 2)),
                                  dtype = torch.float32 , requires_grad = False,device = torch_device)

    def forward(self, x):
        ## convert latent space variables into observed variables
        ldj  = 0
        logq = self.distribution.log_prob(x)
        for i in range(self.num_coupling_layers):
            s,t = self.layers_list[i](x*self.masks[i])
            y   = self.masks[i]*x + (1-self.masks[i])*(x*torch.exp(s) + t)        
            ldj+= torch.sum((1 - self.masks[i])*s, dim = [1,2,3])
            x = y
        logq -= ldj
        return y, logq

    def inverse(self, y):
        ## convert observed variables into latent space variables        
        ldj = 0
        for i in reversed(range(self.num_coupling_layers)):
            s,t = self.layers_list[i](y*self.masks[i])
            x = self.masks[i]*y + (1-self.masks[i])*((y - t)*torch.exp(-s))
            ldj += torch.sum((1 - self.masks[i])*(-s), dim = [1,2,3])
            y = x
        logq  = self.distribution.log_prob(x)
        logq += ldj
        return x, logq

    def log_pdf(self,y):
        ldj  = 0
        for i in reversed(range(self.num_coupling_layers)):
            s,t = self.layers_list[i](y*self.masks[i])
            x = self.masks[i]*y + (1-self.masks[i])*((y - t)*torch.exp(-s))
            ldj += torch.sum((1 - self.masks[i])*(-s), dim = [1,2,3])
            y = x
        logq  = self.distribution.log_prob(x)
        logq += ldj
        return logq
        
    
    def jacobian(self, x):
        logq = self.log_pdf(x)
        v = torch.zeros_like(logq)
        v[:] = 1.
        dy_dx = autograd.grad(logq, x, grad_outputs=v, retain_graph=True,
                                create_graph=True, allow_unused=True)[0]  # shape [B, N]
        
        jacobian = dy_dx.requires_grad_()
        return jacobian
      

def serial_sample_generator(model, dist, batch_size, N_samples):
    x, logq, logp = None, None, None
    
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
        # we're out of samples to propose, generate a new batch
            #z = base_dist.sample_n(batch_size)
            z = model.distribution.sample_n(batch_size)
            model.eval()
            x, logq = model(z)
            logp = dist(x)
        yield x[batch_i], logq[batch_i], logp[batch_i]
        
def make_mcmc_ensemble(model, dist, batch_size, N_samples,seed = 1000):
    rs = np.random.RandomState(seed = seed)
    # build Markov chain
    history = {'x' : [],'logq' : [],'logp' : [],'accepted' : [],'samples' : [] }
    sample_gen = serial_sample_generator(model,dist, batch_size, N_samples)
    with torch.no_grad(): 
        for new_x, new_logq, new_logp in sample_gen:
            if len(history['logp']) == 0:
                # always accept first proposal, Markov chain must start somewhere
                accepted = True

            else:
                # Metropolis acceptance condition
                last_logp = history['logp'][-1]
                last_logq = history['logq'][-1]
                p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
                p_accept = min(1, p_accept)
                draw = rs.rand() # ~ [0,1]
                if draw < p_accept:
                    accepted = True
                else:
                    accepted = False
                    new_x = history['x'][-1]
                    new_logp = last_logp
                    new_logq = last_logq
                    # Update Markov chain
            history['logp'].append(new_logp)
            history['logq'].append(new_logq)
            history['x'].append(new_x)
            history['accepted'].append(accepted)
            history['samples'].append(new_x.detach().cpu().numpy())
            
    

    return history
    

def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg
 