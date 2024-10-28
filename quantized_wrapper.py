# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:46:03 2024

@author: 7000028246
"""
from torch import nn, autograd
import numpy as np
from matplotlib import pyplot as plt

def reorder_by_ch(in_w, channel_axis = 0):
    order_array = [channel_axis, *[i for i in range(len(in_w.shape)) if i != channel_axis]]
    w_perm = np.transpose(in_w, order_array)        
    n_channels = w_perm.shape[0]
    return w_perm.reshape((n_channels, -1))
    

def my_q(w, th, bits):
    fc = 2**(bits-1)
    return np.minimum(np.maximum(np.round(w/th*fc), -fc), fc-1)*th/fc


def search_weights_threshold(in_w, bits_array, channel_axis = 0):
    ########################
    # Reorder
    ########################
    w_reshape = reorder_by_ch(in_w, channel_axis)
    n_channels = w_reshape.shape[0]
            
    th_m = np.abs(w_reshape).max(axis=1)
    th_m = np.reshape(th_m, (-1, 1))
    w_reshape1 = np.reshape(w_reshape, w_reshape.shape + (1,))

    th_arr = []
    err_arr = []
    for bits in bits_array:
        th_h = th_m
        S = np.linspace(0.1, 1.5,100)
        th_S = np.reshape(th_h*S, (n_channels,1,-1))
        e = my_q(w_reshape1, th_S, bits)-w_reshape1
        es1 = (e*e).mean(axis=1)
        es1 = es1.T 
    
        am = np.argmin(es1, axis=0)
        em = np.min(es1, axis=0)
    
        th_h = th_h*np.reshape(S[am], (-1, 1))
        
        S = np.linspace(0.8, 1.2,100)
        th_S = np.reshape(th_h*S, (n_channels,1,-1))
        e = my_q(w_reshape1, th_S, bits)-w_reshape1
        es1 = (e*e).mean(axis=1)
        es1 = es1.T 
        plt.plot(es1)
    
        am = np.argmin(es1, axis=0)
        em = np.min(es1, axis=0)
    
        th_h = th_h*np.reshape(S[am], (-1, 1))

        th_arr.append(th_h)
        err_arr.append(em)
    
    return th_arr, err_arr

class QuantizedWrapper(nn.Module):
    def __init__(self, in_op, name: str):
        super().__init__()

        if not (isinstance(in_op, nn.Conv2d) or isinstance(in_op, nn.Linear)):
            raise Exception(f"Unknown Operations Type{type(in_op)}")

        # Module
        self.add_module("weight_op", in_op)
        self.name = name

        # Quantization
        self.w_n_bits = 8
        self.w_threshold = None

        # Hessian info
        self.image_ix = None
        self.image_cnt = 0
        self.per_image_h_info = dict()
        
        
        in_w = self.weight_op.weight.detach().numpy()                
        
        # print(in_w.shape)
        if isinstance(in_op, nn.Conv2d): 
            assert len(in_w.shape) == 4
        else:
            assert isinstance(in_op, nn.Linear)
            assert len(in_w.shape) == 2

    def forward(self, x):
        return self.weight_op(x)
    
    def finalize_derivative(self, image_ix = None):
        assert self.image_cnt > 0
        assert self.image_ix == image_ix
        
        w_reshape = reorder_by_ch(self.per_layer_lfh_sum/self.image_cnt, channel_axis = 0)
        
        self.per_image_h_info[self.image_ix] = w_reshape.sum(axis=1)
        self.image_cnt = 0
        self.image_ix = None
            
    
    def update_derivative_info(self, outputs, image_ix = None): 
    
        jac_v = autograd.grad(outputs=outputs, 
                              inputs=self.op.weight,
                              retain_graph=True)[0]
                
        per_layer_lfh = (jac_v ** 2).detach().numpy()
        
        if self.image_ix == image_ix:
            self.per_layer_lfh_sum += per_layer_lfh
            self.image_cnt += 1
        else:
            self.per_layer_lfh_sum = per_layer_lfh
            self.image_ix = image_ix
            self.image_cnt = 1
    
    
    
class ActivationQuantizedWrapper(nn.Module):
    def __init__(self, in_op, name: str):
        super().__init__()
        
        if in_op is not None:
            self.add_module("act_op", in_op)
        else:
            self.act_op = None
            print('name has no implemetation:', name)
            assert False
            
        # Module
        self.name = name

        self.image_ix = None
        self.image_cnt = 0
        self.per_image_h_info = dict()
    def forward(self, x):
        self.x_tensor = self.act_op(x)    
        self.x_tensor_mean = self.x_tensor.mean().item()
        return self.x_tensor
    
    def finalize_derivative(self, image_ix = None):
        assert self.image_cnt > 0
        assert self.image_ix == image_ix
        
        self.per_layer_lfh_sum = self.per_layer_lfh_sum/self.image_cnt
        self.per_image_h_info[self.image_ix] = self.per_layer_lfh_sum.sum()
        self.image_cnt = 0
        self.image_ix = None
            
    
    def update_derivative_info(self, outputs, image_ix = None): 
    
        jac_v = autograd.grad(outputs=outputs, 
                              inputs=self.x_tensor,  
                              retain_graph=True)[0]
        
        assert jac_v.shape[0] == 1 # single in batch
        
        per_layer_lfh = (jac_v ** 2).detach().numpy()
        
        if self.image_ix == image_ix:
            self.per_layer_lfh_sum += per_layer_lfh
            self.image_cnt += 1
        else:
            self.per_layer_lfh_sum = per_layer_lfh
            self.image_ix = image_ix
            self.image_cnt = 1
                        
            
    