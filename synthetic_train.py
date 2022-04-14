# -*- coding: utf-8 -*-
"""
Created on Aug 6 11:33:45 2021

@author: liyan
"""
import pandas as pd
import numpy as np
# import networkx as nx
import random
import copy
from random import sample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

## Simulation: 9 variables for classification problem, 8 variables for regression problem
cla_data = []
reg_data = []
for i in range(9):
    cla_df = pd.read_csv("simulate500_cla%d.csv"%(i))
    cla_data.append(cla_df)
    reg_df = pd.read_csv("simulate500_reg%d.csv"%(i))
    reg_data.append(reg_df)    
    
## For the selected 10 domains
#from DAPCASBayes import DAPDAG_Bayes, DoAMLP_Bayes
#from DoACASTLE import DoACASTLE, DoAMLP
from DOCASTLE import DOCASTLE

from DAPDAGBayes import DAPDAG_Bayes_RG
from utils.graph import draw_graph, clip_graph, adjacency

reg_data = []
for i in range(10):
    reg_df = pd.read_csv("data/simulate500_reg%d.csv"%(i))
    reg_data.append(reg_df)

reg_var = list(reg_df.columns)
n_hidden = 8
e_out = 1

mlp_r2 = []
dam_r2 = []
damb_r2 = []
cas_r2 = []
dap_r2 = []
dapb_r2 = []

# true graph of observed data
true_graph = np.array([[0, 1, 0, 1, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0]])
draw_graph(true_graph, reg_var,"true")

import torch as th
skeleton = th.cat([th.tensor(true_graph).T,th.tensor([1, 0, 1, 1, 1, 0, 1, 1]).view(8,1)],dim=1) 

dapbr2_SHD = []
pred_Es = []

for i in range(10):

    reg_copy = copy.deepcopy(reg_data)
    domains = list(range(10))
    ## select the target domain and source domains

    reg_target = reg_copy[i]
    reg_source = [reg_copy[m] for m in domains if m!=i]
    reg_merged = pd.concat(reg_source)
    print("target domain: "+ str(i))
    
    # DAPDAG-RG
    dapb2reg = DAPDAG_Bayes_RG(n_vars=8,nb_con=8,nh=16,hlayers=1,skeleton=skeleton,
                               e_out=1,f_hlayers=2,g_hlayers=1,f_hidden=32,g_hidden=16,reg=True)
    dapb2reg.fit(reg_source,train_epochs=1000,batch_size=1,lr=.001,
                 alpha=1,beta=1,dagstart=0.2,tol=50,power=20)
    dapb2reg_pred = dapb2reg.predict(reg_target.iloc[:,:])
    r2 = r2_score(reg_target.iloc[:,-1],dapb2reg_pred)
    
    pred_Es.append(dapb2reg.get_E(reg_target.iloc[:,:]))
    
    # return the learned graph
    graph4 = dapb2reg.get_graph()
    draw_graph(clip_graph(graph4, 0.5), reg_var,"graph_{}".format(i))
    
    shd = np.sum(np.abs(adjacency(graph4, 0.5)-true_graph))
    dapbr2_SHD.append((r2,shd))
    '''
    ## MLP
    mlpreg = MLPRegressor(hidden_layer_sizes=(n_hidden,n_hidden),max_iter=2000,
                              activation ='relu',solver='adam',validation_fraction=0.1,
                              early_stopping=True,random_state=2021,verbose=True)
        
    mlpreg.fit(reg_merged.iloc[:,:-1], reg_merged.iloc[:,-1])
    mlpreg_pred = mlpreg.predict(reg_target.iloc[:,:-1])
    mlp_r2.append(r2_score(reg_target.iloc[:,-1],mlpreg_pred))
    
    # doamlp
    damreg = DoAMLP(n_vars=8,nh=8,f_hidden=8,f_hlayers=1,
                    g_hlayers=1,g_hidden=8,reg=True)
    damreg.fit(reg_source,train_epochs=500,batch_size=1,
               beta=0.2,gamma=0.1,e_reg=0.5,tol=10)
    damreg_pred = damreg.predict(reg_target.iloc[:,:])
    dam_r2.append(r2_score(reg_target.iloc[:,-1],damreg_pred))
    
    
    dambreg = DoAMLP_Bayes(n_vars=8,nh=16,f_hidden=16,f_hlayers=1,
                    g_hlayers=1,g_hidden=16,reg=True)
    dambreg.fit(reg_source,train_epochs=500,batch_size=1,
                beta=0.1,gamma=0.05,tol=20)
    dambreg_pred = dambreg.predict(reg_target.iloc[:,:])
    damb_r2.append(r2_score(reg_target.iloc[:,-1],dambreg_pred))
    
    ## CASTLE
    casreg = DOCASTLE(n_vars=8,nb_con=8,nh=16,reg=True)
    casreg.fit(reg_merged,train_epochs=500,alpha=1,
               pho=0.5,beta=0.1,tol=20,power=20)
    casreg_pred = casreg.predict(reg_target.iloc[:,:])
    cas_r2.append(r2_score(reg_target.iloc[:,-1],casreg_pred))
    graph1 = casreg.get_graph()
    print(graph1)
    ## build model of DAPDAS
    dapreg = DoACASTLE(n_vars=8,nb_con=8,nh=16,f_hlayers=1,
                       g_hlayers=1,f_hidden=16,g_hidden=16,reg=True)
    dapreg.fit(reg_source,train_epochs=500,batch_size=1,lambd=1,alpha=1,
               gamma=0.05,beta=0.05,e_reg=0.05,tol=20,power=20)
    dapreg_pred = dapreg.predict(reg_target.iloc[:,:])
    dap_r2.append(r2_score(reg_target.iloc[:,-1],dapreg_pred))
    graph2 = dapreg.get_graph()
    print(graph2)
    
    # DAPDAG
    dapbreg = DAPDAG_Bayes(n_vars=8,nb_con=8,nh=16,f_hlayers=1,
                       g_hlayers=1,f_hidden=16,g_hidden=16,reg=True)
    dapbreg.fit(reg_source,train_epochs=500,batch_size=1,lambd=0.1,alpha=5,
               gamma=0.05,beta=0.1,tol=50,power=20)
    dapbreg_pred = dapbreg.predict(reg_target.iloc[:,:])
    dapb_r2.append(r2_score(reg_target.iloc[:,-1],dapbreg_pred))
    graph3 = dapbreg.get_graph()
    graph3 = clip_graph(graph3,0.5)
    draw_graph(graph3, reg_var,"graph_{}".format(i))
    '''
