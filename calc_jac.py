# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
import torch
import copy
import numpy as np
from torch.fx import symbolic_trace
from imagenet_representative_dataset import get_representative_dataset
from fx_utils import my_Fx, pre_hook, hook

import matplotlib.pyplot as plt
import timm

def add_soft_max(tensor_dict):
    tensor_dict['output-softmax'] = torch.nn.functional.softmax(torch.Tensor(tensor_dict['output']), dim=1)

def showim(I):
    m = I.min()
    M = I.max()
    plt.imshow(np.transpose((I-m)/(M-m), axes=(1,2,0)))
    plt.show()



## New code - replace q

LINEAR_OPS = [(torch.nn.Conv1d,),
              (torch.nn.Conv2d,),
              (torch.nn.Conv3d,),
              (torch.nn.Linear,)]

ACTIVATION_OPS = [(torch.nn.ReLU,),
                  (torch.nn.ReLU6,),
                  (torch.nn.Identity,)]

from torch import fx,nn

from quantized_wrapper import QuantizedWrapper, ActivationQuantizedWrapper


def _matches_module_pattern(pattern, node, in_node, modules):
    # if len(node.args) == 0:
    #     return False
    nodes = (in_node, node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def _parent_name(target):
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def _replace_node_module(node, modules, new_module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)
    return new_module

def replace2quantized_model(in_model, skip_layers = [], linear_patterns=LINEAR_OPS, 
                            act_patterns=ACTIVATION_OPS):
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    in_model = in_model
        
    fx_model = fx.symbolic_trace(in_model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in linear_patterns:
        for node in new_graph.nodes:
            for in_node in node.args:
                if _matches_module_pattern(pattern, node, in_node, modules):
                    # if in_node.target in skip_layers:
                        # continue
                    target_op = modules[in_node.target]

                    wrap_node = _replace_node_module(in_node, modules, QuantizedWrapper(target_op, name=in_node.target))
                    
                    succs_nodes = [m for m in new_graph.nodes if in_node in m.args] + [in_node.next]
                    # (1) If this node is the last linear op then we are not quantizing its activation
                    # (2) if there is a conv -> relu in the graph then we only quantize the relu activation output,
                    #     otherwise, we wrap the convolution with a weights quantizer wrapper (QuantizedWrapper)
                    #     and activation quantizer wrapper (ActivationQuantizedWrapper) on top of it.
                    if (any(['add' in s.name 
                             or isinstance(modules.get(s.target), (nn.ReLU, nn.ReLU6)) 
                             or 'downsample' in s.name 
                             for s in succs_nodes]) ):
                        continue
                    else:
                        wrap_node = _replace_node_module(in_node, modules,ActivationQuantizedWrapper(wrap_node, name=in_node.target))
                        print(wrap_node)
                        
    for pattern in act_patterns:
        for node in new_graph.nodes:
            for in_node in node.args:
                if _matches_module_pattern(pattern, node, in_node, modules):
                    target_op = modules[in_node.target]
                    wrap_node = _replace_node_module(in_node, modules, ActivationQuantizedWrapper(target_op, name=in_node.target))
                    print(wrap_node)

    return fx.GraphModule(fx_model, new_graph)



from torch.fx.experimental.optimization import fuse

class QuantizationModelWrapper:
    def __init__(self, in_model):
        print("Start BN Start")
        model_fold = fuse(in_model)
        print("End BN Fuse")
        print("Starting Layer Wrapping")
        
        last_linear_layer = [n for n, m in model_fold.named_modules()
                             if isinstance(m, tuple([t[0] for t in LINEAR_OPS]))][-1]
        
        self.qmodel = replace2quantized_model(model_fold, skip_layers = [last_linear_layer])

        self.qmodel.train(False)

        print("End Layer Wrapping")

    def __call__(self, x):
        return self.qmodel(x)

    def apply_weights_quantization(self):
        print("Apply Quantization")
        for n, m in self.qmodel.named_modules():
            if isinstance(m, QuantizedWrapper):
                m.apply_weights_quantization()


timm_nets = ['tinynet_d.in1k','lcnet_075.ra2_in1k','mobilenetv3_small_075.lamb_in1k','lcnet_050.ra2_in1k','mobilenetv3_small_100.lamb_in1k','lcnet_100.ra2_in1k','mobilenetv3_small_050.lamb_in1k','tinynet_e.in1k','rexnet_100.nav_in1k','mobileone_s1.apple_in1k','tinynet_c.in1k','tinynet_b.in1k','hardcorenas_a.miil_green_in1k','hardcorenas_b.miil_green_in1k','efficientnet_b0.ra_in1k', 'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', 'efficientnet_b1.ra4_e3600_r240_in1k']
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# model = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
# model = timm.create_model(timm_nets[0], pretrained=True)


model = model.eval()


qm = QuantizationModelWrapper(model)

act_w = {n:m for n, m in qm.qmodel.named_modules() if isinstance(m, ActivationQuantizedWrapper) }
weight_w = {n:m for n, m in qm.qmodel.named_modules() if isinstance(m, QuantizedWrapper) }

# Get representative dataset generator
representative_dataset_gen = get_representative_dataset('C:/GIT/val_data_imagenet', do_unsqueeze = True, n_iter=4)
rep = representative_dataset_gen()

#ToDo: get also label and validate accuracy

for image_ix, image in enumerate(representative_dataset_gen()):
    print('image:', image_ix)
    # image = next(rep)
    # image_ix = 1
    r = qm(image)
    r_SM = nn.LogSoftmax(dim=1)(r)
    
    for t in range(10):
        print('Hatchison: ',t)
        v = 1-2*torch.randint_like(r_SM, high=2)
        l = torch.mean(r_SM.unsqueeze(dim=1) @ v.unsqueeze(dim=-1))
        for n,m in act_w.items():
            m.update_derivative_info(outputs=l, image_ix = image_ix)
        for n,m in weight_w.items():
            m.update_derivative_info(outputs=l, image_ix = image_ix)

    for n,m in act_w.items():
        m.finalize_derivative(image_ix = image_ix)
    for n,m in weight_w.items():
        m.finalize_derivative(image_ix = image_ix)

for n,m in act_w.items():
    print('*'*50)
    print('Act Hessian: ', n)
    print('*'*50)
    
    print(m.per_image_h_info, np.mean(list(m.per_image_h_info.values())))
    # print(np.mean([x[0] for x in m.per_image_h_info.values()]), np.mean([x[1] for x in m.per_image_h_info.values()]))

for n,m in weight_w.items():
    print('*'*50)
    print('Weight Hessian: ', n)
    print('*'*50)
    
    # print(m.per_image_h_info)
    print(m.per_image_h_info, np.mean(list(m.per_image_h_info.values())))
    # print(np.mean([x[0] for x in m.per_image_h_info.values()]), np.mean([x[1] for x in m.per_image_h_info.values()]))



# image = next(rep)
# image_ix = 2
# r = qm(image)

# for _ in range(40):
#     v = 1-2*torch.randint_like(r, high=2)
#     l = torch.mean(r.unsqueeze(dim=1) @ v.unsqueeze(dim=-1))
#     for n,m in act_w.items():
#         m.update_derivative_info(outputs=l, image_ix = image_ix)


# for n,m in act_w.items():
#     # print(n, m.x_tensor.shape[1]*m.x_tensor.shape[2]*m.x_tensor.shape[3])
#     print(n, m.x_tensor.shape)


# layer = 'features.0.2'
# layer = 'features.7.conv.1.2'
# eps = 1e-2
# act_w[layer].set_eps(eps)


# l1 = torch.mean(r.unsqueeze(dim=1) @ v.unsqueeze(dim=-1))

# act_w[layer].set_eps(-eps)
# r = qm(image.unsqueeze(0))
# l2 = torch.mean(r.unsqueeze(dim=1) @ v.unsqueeze(dim=-1))

# act_w[layer].set_eps(0.0)
# r = qm(image.unsqueeze(0))

# jac_v = torch.autograd.grad(outputs=l, 
#                       inputs=act_w[layer].x_tensor,  # Change 1: take derivative w.r.t to weights 
#                       retain_graph=True)[0]

# print(jac_v.sum(), (l1-l2)/eps/2.0, act_w[layer].x_tensor.abs().mean())

raise KeyError()


        






for n, m in qm.qmodel.named_modules():
    if isinstance(m, QuantizedWrapper):
        jac_v = torch.autograd.grad(outputs=l, 
                              inputs=m.op.weight,  # Change 1: take derivative w.r.t to weights 
                              retain_graph=True)[0]
        print(jac_v)

for n, m in qm.qmodel.named_modules():
    if isinstance(m, ActivationQuantizedWrapper):
        
        print('is eq:', m.x_tensor.mean().item(), m.x_tensor_mean, hasattr(m.act_op, 'inplace'), m.act_op)
        
        jac_v = torch.autograd.grad(outputs=l, 
                              inputs=m.x_tensor,  # Change 1: take derivative w.r.t to weights 
                              retain_graph=True)[0]

