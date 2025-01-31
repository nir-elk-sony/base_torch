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

def replace2quantized_model(in_model, linear_patterns=LINEAR_OPS, 
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
        
        self.qmodel = replace2quantized_model(model_fold)

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

# Get representative dataset generator


representative_dataset_gen = get_representative_dataset('C:/GIT/val_data_imagenet')

image = next(representative_dataset_gen())

my_fx = my_Fx(model)

tensor_dict = my_fx.forward(image)

if False:
    pre_hook1 = lambda model, inp: pre_hook(model, inp, tensor_dict)
    hook1 = lambda model, inp, out: hook(model, inp, out, tensor_dict)

    for k,m in model.named_modules():
        m.register_forward_pre_hook(pre_hook1)
        m.register_forward_hook(hook1)

if False:
    ref_out = model(image.clone().unsqueeze(0))
    assert (tensor_dict['output'] == ref_out).all().item()
    
    ref_out = my_fx.fx_model(image.clone().unsqueeze(0))
    assert (tensor_dict['output'] == ref_out).all().item()
    
    raise KeyError()




add_soft_max(tensor_dict)

std = np.sqrt((image*image).mean().item())

inv_noise_level = [256, 64, 16, 4]

noises = [ torch.randn(image.shape)*std/n_level for n_level in inv_noise_level]


tensors_dict_with_noise = []
for n in noises:
    tensors_dict_with_noise.append(my_fx.forward(image+n))
    showim(image+n)
    add_soft_max(tensors_dict_with_noise[-1])

for compute_node in my_fx.compute_order:
    nominal = tensor_dict.get(compute_node)
    if type(nominal) not in [torch.Tensor, np.ndarray]:
        continue
    errs = []
    for T in tensors_dict_with_noise:
        if False:
            e = (nominal-T[compute_node])/nominal
            e[np.isnan(e)] = 0.0
            err = 1.0/np.abs(e).mean()
        else:
            e = (nominal-T[compute_node])
            err = 1.0/np.sqrt( (e*e).mean().item() / (nominal*nominal).mean().item() )
        errs.append(err)

    des = my_fx.get_layer_desc(compute_node)
    ref = max(errs)
    if not np.isinf(ref) and not np.isnan(ref):
        print(f'{compute_node:40}:', "".join([ f'{round(r):6}({round(ref/r*10)/10:3})' for r in errs ]), des)


print(tensor_dict['output-softmax'].max(), tensor_dict['output-softmax'].argmax())
for T in tensors_dict_with_noise:
    print(T['output-softmax'].max(), T['output-softmax'].argmax())





model_name = type(model).__name__
torch.onnx.export(model,               # model being run
                  image.unsqueeze(0),                         # model input (or a tuple for multiple inputs)
                  f"{model_name}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  #opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


# layer = 'features_1_0_block_1_scale_activation'
# T0 = tensor_dict[layer]
# Te = tensors_dict_with_noise[0][layer]

raise KeyError()




err_sum = {k:np.zeros(m.shape) for k,m in tensor_dict.items()}







for ix in range(50):
    print(ix)
    noise = torch.randn(image.shape)*std/256
    tensor_dict_with_noise = my_fx.forward(image+noise)
    for k,t in err_sum.items():
        ee = tensor_dict_with_noise[k]-tensor_dict[k]
        t += ee*ee

def get_layer_desc(self, compute_node):
    op = self.nodes[compute_node]
    if op.op == 'call_module':
        s = str(self.mods_fx[op.target])
    elif op.op == 'call_function':
        s = op.target.__name__
    else:
        s = ''
    return s

def showim(I):
    m = I.min()
    M = I.max()
    plt.imshow(np.transpose((I-m)/(M-m), axes=(1,2,0)))

image = next(representative_dataset_gen())
showim(image)
noise = torch.randn(image.shape)*std/4
showim(image+noise)



for k in err_sum.keys():
    des = get_layer_desc(my_fx, k)
    # if 'Conv2d' in des:
    print(k, des)
    
layer = 'features_0_0'
layer = 'features_16_conv_0_1'
layer = 'features_15_conv_0_1'
# layer = 'features_1_conv_0_0'
plt.hist(np.reshape(err_sum[layer], (-1,)))
plt.plot(sorted(np.reshape(err_sum[layer], (-1,))))
# type(my_fx.get_layer_desc(layer))
plt.show() 


rel_err = np.abs(tensor_dict[layer]/err_sum[layer])
rel_err[rel_err > 1e9]=0
plt.hist(np.reshape(np.abs(rel_err), (-1,)), bins=100)
raise KeyError()





