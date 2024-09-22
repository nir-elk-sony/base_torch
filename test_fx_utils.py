# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch
import copy
import numpy as np
from torch.fx import symbolic_trace
from imagenet_representative_dataset import get_representative_dataset
from fx_utils import my_Fx        

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()

# Get representative dataset generator
representative_dataset_gen = get_representative_dataset('C:/GIT/CIFAR10/val_data')

image = next(representative_dataset_gen())

my_fx = my_Fx(model)

tensor_dict = my_fx.forward(image)

ref_out = my_fx.fx_model(image.clone().unsqueeze(0))
assert (tensor_dict['output'] == ref_out).all().item()

ref_out = model(image.clone().unsqueeze(0))
assert (tensor_dict['output'] == ref_out).all().item()



std = np.sqrt((image*image).mean().item())

noise1 = torch.randn(image.shape)*std/256
tensor_dict_with_noise1 = my_fx.forward(image+noise1)

noise2 = torch.randn(image.shape)*std/64
tensor_dict_with_noise2 = my_fx.forward(image+noise2)

noise3 = torch.randn(image.shape)*std/8
tensor_dict_with_noise3 = my_fx.forward(image+noise3)

tensor_dict['output-softmax'] = torch.nn.functional.softmax(tensor_dict['output'], dim=1)
tensor_dict_with_noise1['output-softmax'] = torch.nn.functional.softmax(tensor_dict_with_noise1['output'], dim=1)
tensor_dict_with_noise2['output-softmax'] = torch.nn.functional.softmax(tensor_dict_with_noise2['output'], dim=1)
tensor_dict_with_noise3['output-softmax'] = torch.nn.functional.softmax(tensor_dict_with_noise3['output'], dim=1)


for compute_node in my_fx.compute_order:
    nominal = tensor_dict[compute_node]
    e = (nominal-tensor_dict_with_noise1[compute_node])
    err1 = 1.0/np.sqrt( (e*e).mean().item() / (nominal*nominal).mean().item() )
    e = (nominal-tensor_dict_with_noise2[compute_node])
    err2 = 1.0/np.sqrt( (e*e).mean().item() / (nominal*nominal).mean().item() )

    e = (nominal-tensor_dict_with_noise3[compute_node])
    err3 = 1.0/np.sqrt( (e*e).mean().item() / (nominal*nominal).mean().item() )
    
    op = my_fx.nodes[compute_node]
    if op.op == 'call_module':
        s = my_fx.mods_fx[op.target]
    else:
        s = ''
     
    # print(compute_node + ' '*(50-len(compute_node)), round(err1), round(err2), round(err1/err2*100))
    print(f'{compute_node:30}: {round(err1):4} {round(err2):4} {round(err3):4}  {round(err1/err2*100)/100:4}', s)


print(tensor_dict['output-softmax'].max(), tensor_dict['output-softmax'].argmax())
print(tensor_dict_with_noise1['output-softmax'].max(), tensor_dict_with_noise1['output-softmax'].argmax())
print(tensor_dict_with_noise2['output-softmax'].max(), tensor_dict_with_noise2['output-softmax'].argmax())
print(tensor_dict_with_noise3['output-softmax'].max(), tensor_dict_with_noise3['output-softmax'].argmax())

