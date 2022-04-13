# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:09:51 2021

@author: liyan
"""

"""DAG Constraint Loss.
Paper: DAGs with NO TEARS: Smooth Optimization for Structure Learning
Authors: Zheng, Xun; Aragam, Bryon; Ravikumar, Pradeep; Xing, Eric P.
Implementation by Diviyan Kalainathan
"""
import math
import torch as th
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

## Clip and Draw the learned graph
def clip_graph(adj_matrix, thres):
    return (np.abs(adj_matrix) > thres)*adj_matrix

def adjacency(adj_matrix, thres):
    return np.array(adj_matrix > thres)
    

def draw_graph(adj_matrix, all_var, name):
    G = nx.from_numpy_matrix(np.matrix(adj_matrix), create_using=nx.DiGraph)
    pos = nx.shell_layout(G)
    lab_node = dict((i, var) for i,var in enumerate(all_var))
    nx.draw(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G,pos, labels=lab_node, font_size=10, 
                            font_family='sans-serif')
    # plt.savefig('%s.pdf'%(name))
    plt.show()

class SimpleMatrixConnection(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""

    def __init__(self, graph_size, mask=None):
        super(SimpleMatrixConnection, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.weights = th.nn.Parameter(th.FloatTensor(*self.graph_size))
        self.weights.data.normal_()

        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask) == bool and not mask):
            self.register_buffer("mask", mask)

        ones_tensor = th.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = th.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)

    def forward(self):
        """Return a fixed graph."""

        sample_soft = th.sigmoid(self.weights)

        sample_hard = th.where(self.weights > 0, self.ones_tensor, self.zeros_tensor)
        sample = sample_hard - sample_soft.data + sample_soft

        if hasattr(self, "mask"):
            return self.mask * sample
        else:
            return sample

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(self.weights) * self.mask
        else:
            return th.sigmoid(self.weights)


class GraphSampler(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, n_noises, gnh, graph_size, mask=None):
        """Init the model."""
        super(GraphSampler, self).__init__()

        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size

        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)

        ones_tensor = th.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = th.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)


        self.register_buffer("noise_graph_sampler", th.Tensor(1, n_noises))

        layers = []
        layers.append(th.nn.Linear(n_noises, gnh))
        #layers.append(th.nn.BatchNorm1d(gnh))
        layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(gnh, gnh))
        #layers.append(th.nn.BatchNorm1d(gnh))
        layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(gnh, gnh))
        #layers.append(th.nn.BatchNorm1d(gnh))
        layers.append(th.nn.LeakyReLU(.2))
        # layers.append(th.nn.Linear(gnh, gnh))
        # layers.append(th.nn.BatchNorm1d(gnh))
        # layers.append(th.nn.LeakyReLU(.2))
        layers.append(th.nn.Linear(gnh, graph_size*graph_size))
        self.layers = th.nn.Sequential(*layers)

        self.reset_parameters()

    def forward(self):

        self.noise_graph_sampler.normal_()

        output_sampler = self.layers(self.noise_graph_sampler).view(*self.graph_size)

        sample_soft = th.sigmoid(output_sampler)
        sample_hard = th.where(output_sampler > 0, self.ones_tensor, self.zeros_tensor)
        
        #print(output_sampler* self.mask)
        #print(sample_soft* self.mask)
        #print(sample_hard* self.mask)

        sample = sample_hard - sample_soft.data + sample_soft

        return sample * self.mask

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.weight.data.normal_()

class MatrixSampler(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, graph_size, mask=None, gumble=False):
        super(MatrixSampler, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.weights = th.nn.Parameter(th.FloatTensor(*self.graph_size))
        self.weights.data.zero_()
        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask): 
            self.register_buffer("mask", mask)
        self.gumble = gumble

        ones_tensor = th.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = th.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)


    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""

        if(self.gumble):

            drawn_proba = gumbel_softmax(th.stack([self.weights.view(-1), -self.weights.view(-1)], 1),
                               tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        else:
            drawn_proba = gumbel_sigmoid(self.weights, self.ones_tensor, self.zeros_tensor, tau=tau, hard=drawhard)
        
        if hasattr(self, "mask"):
            return self.mask * drawn_proba
        else:
            return drawn_proba
    
    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(self.weights) * self.mask
        else:
            return th.sigmoid(self.weights)
        
    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


class MatrixSampler2(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, graph_size, mask=None):
        super(MatrixSampler2, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.weights = th.nn.Parameter(th.FloatTensor(*self.graph_size))
        self.weights.data.zero_()
        self.v_weights = th.nn.Parameter(th.where(th.eye(*self.graph_size)>0, th.ones(*self.graph_size).fill_(1), th.zeros(*self.graph_size))
                                         .repeat(self.graph_size[1], 1, 1)
                                         .transpose(0, 2))
        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)
    
    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""
        # drawn_proba = gumbel_softmax(th.stack([self.weights.view(-1), -self.weights.view(-1)], 1),
        #                        tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        # corr_weights = (drawn_proba.unsqueeze(0) *
        #                 (self.v_weights/ (.5 * self.v_weights.abs().sum(1, keepdim=True)))).sum(0)
        corr_weights = (self.weights.unsqueeze(1) *
                        (self.v_weights/ self.v_weights.abs().sum(1, keepdim=True))).sum(0)
        out_proba = gumbel_softmax(th.stack([corr_weights.view(-1), -corr_weights.view(-1)], 1),
                               tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        if hasattr(self, "mask"):
            return self.mask * out_proba
        else:
            return out_proba

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(2 * self.weights) * self.mask
        else:
            return th.sigmoid(2 * self.weights)

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


class MatrixSampler3(th.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, graph_size, mask=None, gumbel=True, k=None):
        super(MatrixSampler3, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (graph_size, graph_size)
        else:
            self.graph_size = graph_size
        self.k = k if k is not None else self.graph_size[0] - 1
        self.in_weights = th.nn.Parameter(th.FloatTensor(self.graph_size[0], self.k))
        self.out_weights = th.nn.Parameter(th.FloatTensor(self.k, self.graph_size[1]))
        self.in_weights.data.normal_()
        self.out_weights.data.normal_()
        self.gumbel_softmax = gumbel
        if not gumbel:
            ones_tensor = th.ones(*self.graph_size)
            zeros_tensor = th.zeros(*self.graph_size)
            self.register_buffer("ones_tensor", ones_tensor)
            self.register_buffer("zeros_tensor", zeros_tensor)

        if mask is None:
            mask = 1 - th.eye(*self.graph_size)
        if not (type(mask)==bool and not mask):
            self.register_buffer("mask", mask)

    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""
        corr_weights = self.in_weights @ self.out_weights
        if self.gumbel_softmax:
            out_sample = gumbel_softmax(th.stack([corr_weights.view(-1), -corr_weights.view(-1)], 1),
                                       tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        else:
            sample_soft = th.sigmoid(corr_weights)
            sample_hard = th.where(corr_weights > 0,
                                   self.ones_tensor, self.zeros_tensor)
            out_sample = sample_hard - sample_soft.data + sample_soft

        if hasattr(self, "mask"):
            return self.mask * out_sample
        else:
            return out_sample

    def get_proba(self):
        if hasattr(self, "mask"):
            return th.sigmoid(2 * (self.in_weights @ self.out_weights)) * self.mask
        else:
            return th.sigmoid(2 * (self.in_weights @ self.out_weights))

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


def notears_constr(adj_m, max_pow=None):
    """No Tears for binary adjacency matrixes.
    If adj_m is non binary: give adj_m * adj_m as input (Hadamard product)."""
    m_exp = [adj_m]
    if max_pow is None:
        max_pow = adj_m.shape[1]
    while(m_exp[-1].sum() > 0 and len(m_exp) < max_pow):
        m_exp.append(m_exp[-1] @ adj_m/len(m_exp))

    return sum([i.diag().sum() for idx, i in enumerate(m_exp)])

"""Gumble softmax."""
import torch as th

import torch.distributions.relaxed_bernoulli as relaxed_bernoulli
from torch.distributions.transformed_distribution import  TransformedDistribution
from torch.distributions.transforms import SigmoidTransform,AffineTransform

from torch.distributions.uniform import Uniform

def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    return - th.log(eps - th.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + gumbel_noise
    return th.softmax(y / tau, dims-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    Implementation of pytorch.
    (https://github.com/pytorch/pytorch/blob/e4eee7c2cf43f4edba7a14687ad59d3ed61d9833/torch/nn/functional.py)
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: `[batch_size, n_class]` unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if ``True``, take `argmax`, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)

    return th.log(U) - th.log(1-U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return th.sigmoid(y / tau)


def gumbel_sigmoid(logits, ones_tensor, zeros_tensor, tau=1, hard=False):

    shape = logits.size()
 
    y_soft = _sigmoid_sample(logits, tau=tau)

    if hard:

        y_hard = th.where(y_soft > 0.5, ones_tensor, zeros_tensor)

        y = y_hard.data - y_soft.data + y_soft

    else:
    	y = y_soft

    return y