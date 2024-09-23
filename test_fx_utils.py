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

timm_nets = ['tinynet_d.in1k','lcnet_075.ra2_in1k','mobilenetv3_small_075.lamb_in1k','lcnet_050.ra2_in1k','mobilenetv3_small_100.lamb_in1k','lcnet_100.ra2_in1k','mobilenetv3_small_050.lamb_in1k','tinynet_e.in1k','rexnet_100.nav_in1k','mobileone_s1.apple_in1k','tinynet_c.in1k','tinynet_b.in1k','hardcorenas_a.miil_green_in1k','hardcorenas_b.miil_green_in1k','efficientnet_b0.ra_in1k', 'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', 'efficientnet_b1.ra4_e3600_r240_in1k']
# model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
model = timm.create_model(timm_nets[0], pretrained=True)


model = model.eval()


# Get representative dataset generator
representative_dataset_gen = get_representative_dataset('C:/GIT/CIFAR10/val_data')

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





