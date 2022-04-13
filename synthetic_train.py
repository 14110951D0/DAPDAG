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

## Baseline MLP using simple combined data 
repeat = 10
n_hidden = 8

mlp_cla_auc = []
mlp_cla_apr = []
mlp_reg_r2 = []

random.seed(2021)
for M in [4,6,8]:       
    mlp_auc = []
    mlp_apr = []
    mlp_r2 = []    

    for i in range(20):
        cla_copy = copy.deepcopy(cla_data)
        reg_copy = copy.deepcopy(reg_data)
        domains = sample(range(9),M)
        ## select the target domain and source domains
        cla_target = cla_copy[domains[0]]
        cla_source = pd.concat([cla_copy[m] for m in domains if m!=domains[0]])
        
        reg_target = reg_copy[domains[0]]
        reg_source = pd.concat([reg_copy[m] for m in domains if m!=domains[0]])

        ## standardize the continuous data in classification data
        con_data = cla_source[cla_source.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
    
        scaler = StandardScaler()
        scaler.fit(con_data)

        cla_source[cla_source.columns[0]] = scaler.transform(con_data)
        con_data = cla_target[cla_target.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
        cla_target[cla_target.columns[0]] = scaler.transform(con_data)
        
        ## build model
        classifier = MLPClassifier(hidden_layer_sizes=(n_hidden,n_hidden),max_iter=2000,
                              activation ='relu',solver='adam',validation_fraction=0.1,
                              early_stopping=True,random_state=2021,verbose=True)
        classifier.fit(cla_source.iloc[:,:-1], cla_source.iloc[:,-1])
        cla_pred = classifier.predict_proba(cla_target.iloc[:,:-1])
        mlp_auc.append(roc_auc_score(cla_target.iloc[:,-1],cla_pred[:,1]))
        mlp_apr.append(average_precision_score(cla_target.iloc[:,-1],cla_pred[:,1]))
        
        regressor = MLPRegressor(hidden_layer_sizes=(n_hidden,n_hidden),max_iter=2000,
                              activation ='relu',solver='adam',validation_fraction=0.1,
                              early_stopping=True,random_state=2021,verbose=True)
        
        regressor.fit(reg_source.iloc[:,:-1], reg_source.iloc[:,-1])
        reg_pred = regressor.predict(reg_target.iloc[:,:-1])
        mlp_r2.append(r2_score(reg_target.iloc[:,-1],reg_pred))
        
    mlp_cla_auc.append((np.mean(mlp_auc),np.std(mlp_auc))) 
    mlp_cla_apr.append((np.mean(mlp_apr),np.std(mlp_apr)))
    mlp_reg_r2.append((np.mean(mlp_r2),np.std(mlp_r2)))

## CASTLE
from DOCASTLE import DOCASTLE

cas_cla_auc = []
cas_cla_apr = []
cas_reg_r2 = []

random.seed(2021)
for M in [8]:       
    cas_auc = []
    cas_apr = []
    cas_r2 = []    

    for i in range(10):
        cla_copy = copy.deepcopy(cla_data)
        reg_copy = copy.deepcopy(reg_data)
        domains = sample(range(9),M)
        ## select the target domain and source domains
        cla_target = cla_copy[domains[0]]
        cla_source = pd.concat([cla_copy[m] for m in domains if m!=domains[0]])
        
        reg_target = reg_copy[domains[0]]
        reg_source = pd.concat([reg_copy[m] for m in domains if m!=domains[0]])

        ## standardize the continuous data in classification data
        con_data = cla_source[cla_source.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
    
        scaler = StandardScaler()

        cla_source[cla_source.columns[0]] = scaler.fit_transform(con_data)
        con_data = np.asarray(cla_target[cla_target.columns[0]]).reshape(-1,1)
        cla_target[cla_target.columns[0]] = scaler.transform(con_data)
        print(domains)
        ## build model
        classifier = DOCASTLE(n_vars=9,nb_con=1,nh=8)
        classifier.fit(cla_source,train_epochs=500,beta=0.1,tol=10,power=20)
        cla_pred = classifier.predict(cla_target.iloc[:,:])
        cas_auc.append(roc_auc_score(cla_target.iloc[:,-1],cla_pred))
        cas_apr.append(average_precision_score(cla_target.iloc[:,-1],cla_pred))
        
        regressor = DOCASTLE(n_vars=8,nb_con=8,nh=8,reg=True)
        regressor.fit(reg_source,train_epochs=500,beta=0.1,tol=10,power=20)
        reg_pred = regressor.predict(reg_target.iloc[:,:])
        cas_r2.append(r2_score(reg_target.iloc[:,-1],reg_pred))
        
    cas_cla_auc.append((np.mean(cas_auc),np.std(cas_auc))) 
    cas_cla_apr.append((np.mean(cas_apr),np.std(cas_apr)))
    cas_reg_r2.append((np.mean(cas_r2),np.std(cas_r2)))

## DAPCASTION and DoAMLP 
dam_cla_auc = []
dam_cla_apr = []
dam_reg_r2 = []
dap_cla_auc = []
dap_cla_apr = []
dap_reg_r2 = []

#from DoACASTLE import DoACASTLE, DoAMLP
random.seed(2021)
for M in [8]:
    
    dap_auc = []
    dap_apr = []
    dap_r2 = []
    '''
    dam_auc = []
    dam_apr = []
    dam_r2 = []     
    '''
    for number in range(10):
        cla_copy = copy.deepcopy(cla_data)
        reg_copy = copy.deepcopy(reg_data)
        domains = sample(range(9),M)
        ## select the target domain and source domains
        cla_target = cla_copy[domains[0]]
        cla_source = [cla_copy[m] for m in domains if m!=domains[0]]
        reg_target = reg_copy[domains[0]]
        reg_source = [reg_copy[m] for m in domains if m!=domains[0]]

        ## standardize the continuous data in classification data
        merge = pd.concat(cla_source)
        con_data = merge[merge.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
    
        scaler = StandardScaler()
        scaler.fit(con_data)
        
        for data in cla_source:
            con = data[merge.columns[0]]
            con = np.asarray(con).reshape(-1,1)
            data[merge.columns[0]] = scaler.transform(con) 
            
        con_data = cla_target[cla_target.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
        cla_target[cla_target.columns[0]] = scaler.transform(con_data)
        
        print(domains)
        '''
        # doamlp
        damcla = DoAMLP(n_vars=9,nh=8,f_hidden=8,f_hlayers=1,
                        g_hlayers=1,g_hidden=8)
        damcla.fit(cla_source,train_epochs=1000,batch_size=0.5,beta=0.02,
                   gamma=0.1,e_reg=0.01,tol=20)
        damcla_pred = damcla.predict(cla_target.iloc[:,:])
        
        damreg = DoAMLP(n_vars=8,nh=8,f_hidden=8,f_hlayers=1,
                        g_hlayers=1,g_hidden=8,reg=True)
        damreg.fit(reg_source,train_epochs=1000,batch_size=0.5,beta=0.05,
                   gamma=0.1,e_reg=0.01,tol=20)
        damreg_pred = damreg.predict(reg_target.iloc[:,:])
        
        dam_auc.append(roc_auc_score(cla_target.iloc[:,-1],damcla_pred))
        dam_apr.append(average_precision_score(cla_target.iloc[:,-1],damcla_pred))
        dam_r2.append(r2_score(reg_target.iloc[:,-1],damreg_pred))
        '''
        ## build model of DAPDAS
        dapcla = DoACASTLE(n_vars=9,nb_con=1,nh=8,f_hlayers=1,
                           g_hlayers=1,f_hidden=16,g_hidden=16)
        dapcla.fit(cla_source,train_epochs=1000,batch_size=0.5,lambd=1,alpha=0.5,
                   gamma=0.01,beta=0.1,e_reg=0.1,tol=10,power=20)
        dapcla_pred = dapcla.predict(cla_target.iloc[:,:])
        dap_auc.append(roc_auc_score(cla_target.iloc[:,-1],dapcla_pred))
        dap_apr.append(average_precision_score(cla_target.iloc[:,-1],dapcla_pred))
        
        dapreg = DoACASTLE(n_vars=8,nb_con=8,nh=8,f_hlayers=1,
                           g_hlayers=1,f_hidden=16,g_hidden=16,reg=True)
        dapreg.fit(reg_source,train_epochs=1000,batch_size=0.5,lambd=1,alpha=1,
                   gamma=0.01,beta=0.1,e_reg=0.1,tol=10,power=20)
        dapreg_pred = dapreg.predict(reg_target.iloc[:,:])
        dap_r2.append(r2_score(reg_target.iloc[:,-1],dapreg_pred))
        
    dap_cla_auc.append((np.mean(dap_auc),np.std(dap_auc))) 
    dap_cla_apr.append((np.mean(dap_apr),np.std(dap_apr)))
    dap_reg_r2.append((np.mean(dap_r2),np.std(dap_r2)))
    '''
    dam_cla_auc.append((np.mean(dam_auc),np.std(dam_auc))) 
    dam_cla_apr.append((np.mean(dam_apr),np.std(dam_apr)))
    dam_reg_r2.append((np.mean(dam_r2),np.std(dam_r2)))
    '''
## Bayesian Training:
damb_cla_auc = []
damb_cla_apr = []
damb_reg_r2 = []
dapb_cla_auc = []
dapb_cla_apr = []
dapb_reg_r2 = []
    

random.seed(2021)
for M in [8]:
    
    dapb_auc = []
    dapb_apr = []
    dapb_r2 = []
    '''
    damb_auc = []
    damb_apr = []
    damb_r2 = []     
    '''
    for number in range(10):
        cla_copy = copy.deepcopy(cla_data)
        reg_copy = copy.deepcopy(reg_data)
        domains = sample(range(9),M)
        ## select the target domain and source domains
        cla_target = cla_copy[domains[0]]
        cla_source = [cla_copy[m] for m in domains if m!=domains[0]]
        reg_target = reg_copy[domains[0]]
        reg_source = [reg_copy[m] for m in domains if m!=domains[0]]

        ## standardize the continuous data in classification data
        merge = pd.concat(cla_source)
        con_data = merge[merge.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
    
        scaler = StandardScaler()
        scaler.fit(con_data)
        
        for data in cla_source:
            con = data[merge.columns[0]]
            con = np.asarray(con).reshape(-1,1)
            data[merge.columns[0]] = scaler.transform(con) 
            
        con_data = cla_target[cla_target.columns[0]]
        con_data = np.asarray(con_data).reshape(-1,1)
        cla_target[cla_target.columns[0]] = scaler.transform(con_data)
        
        print(domains)
        '''
        # doamlp
        dambcla = DoAMLP_Bayes(n_vars=9,nh=8,f_hidden=16,f_hlayers=1,
                        g_hlayers=1,g_hidden=16)
        dambcla.fit(cla_source,train_epochs=500,batch_size=0.5,
                    beta=0.02, gamma=0.1,tol=10)
        dambcla_pred = dambcla.predict(cla_target.iloc[:,:])
        damb_auc.append(roc_auc_score(cla_target.iloc[:,-1],dambcla_pred))
        damb_apr.append(average_precision_score(cla_target.iloc[:,-1],dambcla_pred))
        
        dambreg = DoAMLP_Bayes(n_vars=8,nh=8,f_hidden=16,f_hlayers=1,
                        g_hlayers=1,g_hidden=16,reg=True)
        dambreg.fit(reg_source,train_epochs=500,batch_size=0.5,
                    beta=0.05,gamma=0.1,tol=10)
        dambreg_pred = dambreg.predict(reg_target.iloc[:,:])
        damb_r2.append(r2_score(reg_target.iloc[:,-1],dambreg_pred))
        '''
        ## build model of DAPDAS
        dapbcla = DAPDAG_Bayes(n_vars=9,nb_con=1,nh=8,f_hlayers=1,
                           g_hlayers=1,f_hidden=16,g_hidden=16)
        dapbcla.fit(cla_source,train_epochs=500,batch_size=1,lambd=1,alpha=0.5,
                   gamma=0.05,beta=0.1,tol=10,power=20)
        dapbcla_pred = dapbcla.predict(cla_target.iloc[:,:])
        dapb_auc.append(roc_auc_score(cla_target.iloc[:,-1],dapbcla_pred))
        dapb_apr.append(average_precision_score(cla_target.iloc[:,-1],dapbcla_pred))
        
        dapbreg = DAPDAG_Bayes(n_vars=8,nb_con=8,nh=8,f_hlayers=1,
                           g_hlayers=1,f_hidden=16,g_hidden=16,reg=True)
        dapbreg.fit(reg_source,train_epochs=500,batch_size=1,lambd=1,alpha=0.5,
                   gamma=0.05,beta=0.1,tol=10,power=20)
        dapbreg_pred = dapbreg.predict(reg_target.iloc[:,:])
        dapb_r2.append(r2_score(reg_target.iloc[:,-1],dapbreg_pred))
        
    dapb_cla_auc.append((np.mean(dapb_auc),np.std(dapb_auc))) 
    dapb_cla_apr.append((np.mean(dapb_apr),np.std(dapb_apr)))
    dapb_reg_r2.append((np.mean(dapb_r2),np.std(dapb_r2)))
    
    damb_cla_auc.append((np.mean(damb_auc),np.std(damb_auc))) 
    damb_cla_apr.append((np.mean(damb_apr),np.std(damb_apr)))
    damb_reg_r2.append((np.mean(damb_r2),np.std(damb_r2)))
    
    
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