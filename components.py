# -*- coding: utf-8 -*-
"""
Created on Sep 17 16:49:30 2021

@author: liyanke
"""
import torch as th
from torch import nn
import torch.nn.functional as F

# Bayesian Encoder for the environmental variable
class Encoder_Bayes(nn.Module):
    def __init__(self, in_features, f_hlayers=2, f_hidden=16,
                 g_hlayers=2, g_hidden=16, out=1, device="cuda"):
        super(Encoder_Bayes, self).__init__()
        self.out = out
        self.device = device
        
        f_layers = []
        f_layers.append(nn.Linear(in_features, f_hidden))
        f_layers.append(nn.ELU())
        for i in range(f_hlayers-1):
            f_layers.append(nn.Linear(f_hidden, f_hidden))
            f_layers.append(nn.ELU())
        f_layers.append(nn.Linear(f_hidden, out))
        self.f_extractor = nn.Sequential(*f_layers).to(self.device)
        
        g_layers = []
        g_layers.append(nn.Linear(in_features, g_hidden))
        g_layers.append(nn.ELU())
        for i in range(g_hlayers-1):
            g_layers.append(nn.Linear(g_hidden,g_hidden))
            g_layers.append(nn.ELU())
        g_layers.append(nn.Linear(g_hidden, out))
        g_layers.append(nn.ReLU())
        self.g_extractor = nn.Sequential(*g_layers).to(self.device)
        self.g_0 = nn.Parameter(th.randn(out),requires_grad=True).to(self.device)
        
        self.add_module('0', self.f_extractor)
        self.add_module('1', self.g_extractor)
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
                
        if hasattr(self.g_0, 'reset_parameters'):
            self.g_0.reset_parameters()
    
    def forward(self, input):
        x = input
        n = x.shape[0]
        x1 = self.f_extractor(x)
        x2 = self.g_extractor(x)
        x0 = self.g_0.exp()
        V = x2.sum(dim=0) - (n-1)/x0
        V = th.sqrt(V*V)
        c = x1*x2
        E = c.sum(dim=0)/V 
        eps = th.randn(n,self.out).to(self.device)
        return E + eps/th.sqrt(V), E, 1/V, x0

# Encoder using transformed mean of representations
class Encoder(nn.Module):
    def __init__(self, in_features, f_hlayers=2, f_hidden=32, 
                 g_hlayers=2, g_hidden=16, out=1, device="cuda"):
        super(Encoder, self).__init__()
        self.out = out
        self.device = device
        
        layers = []
        layers.append(nn.Linear(in_features, f_hidden))
        layers.append(nn.ELU())
        for i in range(f_hlayers-1):
            layers.append(nn.Linear(f_hidden, f_hidden))
            layers.append(nn.ELU())
        self.feature_extractor = nn.Sequential(*layers).to(self.device)
        
        vlayers = []
        vlayers.append(nn.Linear(f_hidden, g_hidden))
        vlayers.append(nn.ReLU())
        for i in range(g_hlayers-1):
            vlayers.append(nn.Linear(g_hidden, g_hidden))
            vlayers.append(nn.ReLU())
        vlayers.append(nn.Linear(g_hidden, out))
        vlayers.append(nn.ReLU())
        self.variance_extractor = nn.Sequential(*vlayers).to(self.device)
        
        regressor = []
        regressor.append(nn.Linear(f_hidden, g_hidden))
        regressor.append(nn.ELU())
        for i in range(g_hlayers-1):
            regressor.append(nn.Linear(g_hidden, g_hidden))
            regressor.append(nn.ELU())
        regressor.append(nn.Linear(g_hidden, out))
        self.regressor = nn.Sequential(*regressor).to(self.device)
        
        self.g_0 = nn.Parameter(th.randn(out),requires_grad=True).to(self.device)
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        self.add_module('2', self.variance_extractor)
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        n = x.shape[0]
        x = self.feature_extractor(x)
        x = x.mean(dim=0)
        v = self.variance_extractor(x) 
        #x = th.cat([x1, x2], -1)
        E = self.regressor(x)
        eps = th.randn(n,self.out).to(self.device)
        
        x0 = self.g_0.exp()
        return E + eps*v, E, v, x0

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'



class Causal_filter(nn.Module):
    def __init__(self, n_input, mask, nh=32):
        """Init the model."""
        super(Causal_filter, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.weights = nn.Parameter(th.randn(n_input,nh)*0.1*mask, requires_grad=True)
        self.biases = nn.Parameter(th.randn(nh)*0.1, requires_grad=True)
        
    def forward(self, X):
        return F.elu(th.matmul(X, self.weights*self.mask)+self.biases)
    
    def get_weights(self):
        return self.weights*self.mask
    
    def reset_parameters(self):                
        if hasattr(self.weights, 'reset_parameters'):
            self.weights.reset_parameters()
        if hasattr(self.biases, 'reset_parameters'):
            self.biases.reset_parameters()


class Shared_layers(nn.Module):
    def __init__(self, nh=32, hlayers=1):
        """Init the model."""
        super(Shared_layers, self).__init__()
        # The intermediate hidden layers are shared
        slayers = []
        for i in range(hlayers):
            slayers.append(nn.Linear(nh,nh))
            slayers.append(nn.Dropout(0.2))
            slayers.append(nn.ELU())     
        
        self.slayers = nn.Sequential(*slayers)
        
    def forward(self, X):
        return self.slayers(X)        
    
    def reset_parameters(self):
        for slayer in self.slayers:
            if hasattr(slayer, 'reset_parameters'):
                slayer.reset_parameters()

class Output_layer(nn.Module):
    def __init__(self, nh=32):
        super(Output_layer, self).__init__()  

        self.out = nn.Linear(nh, 1)
                                   
    def forward(self, X):
        return self.out(X)
    
    def reset_parameters(self):
        self.outputs.reset_parameters()


'''
# MLE Encoder for the environmental variable
class Encoder(nn.Module):
    def __init__(self, in_features, f_hlayers=2, f_hidden=32,
                 g_hlayers=2, g_hidden=32, out=1):
        super(Encoder, self).__init__()
        
        f_layers = []
        f_layers.append(nn.Linear(in_features, f_hidden))
        f_layers.append(nn.ELU())
        for i in range(f_hlayers-1):
            f_layers.append(nn.Linear(f_hidden, f_hidden))
            f_layers.append(nn.ELU())
        f_layers.append(nn.Linear(f_hidden, out))
        
        self.f_extractor = nn.Sequential(*f_layers)
        
        g_layers = []
        g_layers.append(nn.Linear(in_features, g_hidden))
        g_layers.append(nn.ELU())
        for i in range(g_hlayers-1):
            g_layers.append(nn.Linear(g_hidden,g_hidden))
            g_layers.append(nn.ELU())
        g_layers.append(nn.Linear(g_hidden, out))
        g_layers.append(nn.ReLU())
        self.g_extractor = nn.Sequential(*g_layers)
        self.g_0 = nn.Parameter(th.randn(out)*0.1,requires_grad=True)
        
        self.add_module('0', self.f_extractor)
        self.add_module('1', self.g_extractor)
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
                
        if hasattr(self.g_0, 'reset_parameters'):
            self.g_0.reset_parameters()
            
    def forward(self, input):
        x = input
        n = x.shape[0]
        x1 = self.f_extractor(x)
        x2 = self.g_extractor(x)
        x0 = self.g_0
        V = x2.sum(dim=0) - (n-1)*(x0*x0)
        V = th.sqrt(V*V)
        c = x1*x2
        E = c.sum(dim=0)/V 
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'f Exctractor=' + str(self.f_extractor) \
            + '\n g Exctractor=' + str(self.g_extractor) \
            + '\n g_0=' + str(self.g_0)  + ')'

# encoder using mean of representations 
class Encoder(nn.Module):
    def __init__(self, n_input, e_nh=32, e_out=1, e_hlayers=1):
        """Init the model."""
        super(Encoder, self).__init__()
    
        layers = []
        layers.append(nn.Linear(n_input, e_nh))
        layers.append(nn.ELU())
        for i in range(e_hlayers-1):
            layers.append(nn.Linear(e_nh,e_nh))
            layers.append(nn.ELU())
            
        layers.append(nn.Linear(e_nh,e_out))
        layers.append(nn.Tanhshrink())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.layers(X)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
'''
