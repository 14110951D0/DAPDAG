"""DAPCAS Bayes

By Yanke Li for Domain Adaptation
"""
import random
import numpy as np
import torch as th
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.components import Encoder, Causal_filter, Shared_layers, Output_layer #Encoder_Bayes
from utils.graph import SimpleMatrixConnection, MatrixSampler

def kl_loss(mu, var, x0):
    return -0.5 * th.sum(1 + th.log(var) - th.log(x0) - mu*mu/x0 - var/x0) 


class DAPDAG_Bayes_RG(object):
    def __init__(self, n_vars, nb_con, use_self=False, f_hlayers=2, g_hlayers=2, skeleton=None,
                 f_hidden=32, g_hidden=32, hlayers=1, nh=32, e_out=1, device="cuda", reg=False):
        """Init and parametrize the model.
        
        lr: Learning rate
        l1: L1 penalization on the causal filters
        nh: Number of hidden units in the shared hidden layers
        batchsize: Size of the batches to be fed to the model.
        """
        super(DAPDAG_Bayes_RG, self).__init__()
        self.n_vars = n_vars
        self.nb_con = nb_con
        self.nh = nh
        self.reg = reg
        self.e_out = e_out
        self.device = device
        self.skeleton = skeleton,
        
        # Store layers weight & bias

        self.encoder = Encoder(in_features=n_vars-1, f_hlayers=f_hlayers,
                                     g_hlayers=g_hlayers, out = e_out, f_hidden = f_hidden,
                                     g_hidden = g_hidden, device=self.device)
        
        ## building graph sampler including gates for latent noise variables
        
        self.graph = SimpleMatrixConnection((n_vars, n_vars+e_out),mask=skeleton).to(self.device)
        self.graph.weights.data.fill_(1)
        '''
        self.graph = MatrixSampler((n_vars, n_vars+e_out),gumble=False).to(self.device)
        '''
        self.filters = {}
        #self.shared = Shared_layers(nh=nh, hlayers=hlayers).to(self.device)
        self.hidden = {}
        self.outputs = {}
         
        # Create the input and output weight matrix for each feature
        th.manual_seed(2021)        
        for i in range(self.n_vars):
            mask = th.ones(self.n_vars+e_out, nh)
            if not use_self:
                mask[i,:] = 0
            self.filters[str(i)] = Causal_filter(self.n_vars+e_out, mask,nh=nh).to(self.device)
            self.hidden[str(i)] = Shared_layers(nh=nh, hlayers=hlayers).to(self.device)
            self.outputs[str(i)] = Output_layer(nh=nh).to(self.device)

                
        parameters1 = []
        parameters1.extend(list(self.encoder.parameters()))
        
        parameters2 = []
        parameters2.extend(list(self.graph.parameters()))
        
        for i in range(self.n_vars):
            parameters2.extend(list(self.filters[str(i)].parameters()))
            parameters2.extend(list(self.hidden[str(i)].parameters()))
            parameters2.extend(list(self.outputs[str(i)].parameters()))
        
        '''
        parameters.extend(list(self.shared.parameters()))
        for i in range(self.n_vars):
            parameters.extend(list(self.outputs[str(i)].parameters()))
        '''
        self.parameters1 = parameters1
        self.parameters2 = parameters2
    
    def fit(self, domains_data, skeleton=None, beta=0.1, 
            train_epochs=200, batch_size=0.5, lr=.001,
            lambd=1, alpha=1, validation=0.1, tol=10, dagstart=0.2,
            verbose=True, power=None,seed=2021, log=None):
        
        optimizer1 = th.optim.Adam(self.parameters1, lr=lr)
        optimizer2 = th.optim.Adam(self.parameters2, lr=lr)
        
        # training loops
        if verbose:
            pbar = tqdm(range(train_epochs))
        else:
            pbar = range(train_epochs)
        
        criterion1 = nn.MSELoss(reduction='sum')
        criterion2 = nn.BCEWithLogitsLoss(reduction='sum')
        if not self.reg:
            criterion3 = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            criterion3 = nn.MSELoss(reduction='sum')
        
        random.seed(seed)
        train_data = []
        val_data = []
        
        for data in domains_data:
            train, val = train_test_split(data, test_size=validation)
            train_data.append(train)
            val_data.append(val)
        
        domains = list(range(len(train_data)))
        sizes = [data.shape[0] for data in train_data]
        best = th.tensor(1e10).to(self.device)
        count = 0
        va_up = True # flag for updating variational parameters
        
        for epoch in pbar:
            
            for d in domains:
                dom = random.choices(domains,weights=sizes,k=1)[0]
                batchsize = int(np.floor(batch_size*sizes[dom]))
                train = th.tensor(train_data[dom].values.astype('float32')).to(self.device)
                
                for step in range(1, (sizes[dom]//batchsize) + 1):
                    idx = random.sample(range(sizes[dom]), batchsize)
                    data = train[idx,:].to(self.device)
                    
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    # Reconstruction loss and supervised loss
                    
                    reconstruct_loss = 0
                    #supervised_loss = 0
                    
                    E, mu, var, x0 = self.encoder(data[:,:-1])
                    
                    kl = kl_loss(mu, var, x0)
                    drawn_graph = self.graph()
                    
                    for i in range(self.n_vars):
                        output = self.filters[str(i)](th.cat([data,E],1)*drawn_graph[i,:])
                        #output = self.shared(output)
                        output = self.hidden[str(i)](output)
                        output = self.outputs[str(i)](output)
                        
                        if i < self.nb_con:
                            out_loss = 0.5*criterion1(output.squeeze(), data[:,i])
                        else:
                            out_loss = criterion2(output.squeeze(), data[:,i])
                        
                        if self.reg and i == self.n_vars-1:
                            out_loss = 0.5*criterion1(output.squeeze(), data[:,i])
                            supervised_loss = out_loss
                        else:
                            supervised_loss = out_loss
                        
                        reconstruct_loss += out_loss 

                    # l1 norm loss and DAG loss of causal filters
                    L = drawn_graph.sum()
                    
                    ## truncated power series
                    if power is None:
                        power = self.n_vars+1
                
                    coff = 1.0
                    W = self.graph.get_proba()[:,:self.n_vars].t()
                    h = 0        
                    Z_in = th.eye(self.n_vars).to(self.device)
                    for i in range(1,power):   
                        Z_in = th.matmul(Z_in, W) 
                        h += 1./coff * th.trace(Z_in)
                        coff = coff * (i+1)
                    
                    # combine all losses
                    if epoch > train_epochs * dagstart:
                        loss = reconstruct_loss + kl + beta*L + alpha*h*(1+0.5*h)
                    else:
                        loss = reconstruct_loss + kl + beta*L
                    
                    loss.backward(retain_graph=True) 
                    
                    if va_up:
                        optimizer1.step()
                    else:
                        optimizer2.step()
                        
            va_up = not va_up # alternate the maximisation-maximisation procedure            
            
            val_loss = 0 
            val_n = 0
            for val in val_data:
                data = th.tensor(val.values.astype('float32')).to(self.device)
                val_n += val.shape[0]
                val_loss += criterion3(th.tensor(self.predict(val.iloc[:,:])).squeeze().to(self.device),
                                       data[:,-1])
            
            val_loss = val_loss/val_n
            if self.reg:
                val_loss = 0.5*val_loss
            if val_loss < best:
                best = val_loss
                count = 0
            else:
                count += 1
            
            if verbose:
                pbar.set_postfix(recons = reconstruct_loss.item(), 
                                 L = L.item(), kl = kl.item(),
                                 sup = supervised_loss.item(),
                                 h = h.item(),val = val_loss.item())
                    
            if count > tol:
                print("Early stopping")
                break
        
        return 
    
    def get_graph(self):
            
        W = self.graph.get_proba()[:,:self.n_vars].t()
        
        return W.cpu().detach().numpy()
    
   
    def get_E(self, data):
        data = th.tensor(data.values.astype('float32')).to(self.device)    
        E, mu, var, x0 = self.encoder(data[:,:-1])
        
        return (mu.item(),var.item())
        
    
    def predict(self, data):
        data = th.tensor(data.values.astype('float32')).to(self.device)
        n = data.shape[0]
        prediction = 0
        for i in range(n):
            drawn_graph = self.graph()
            E, _, _, _  = self.encoder(data[:,:-1])
            pred = self.filters[str(self.n_vars-1)](th.cat([data,E],1)*drawn_graph[self.n_vars-1,:])
            #pred = self.shared(pred)
            pred = self.hidden[str(self.n_vars-1)](pred)
            pred = self.outputs[str(self.n_vars-1)](pred)
            prediction += pred
            
        prediction = prediction/n
        
        return prediction.cpu().detach().numpy()
        