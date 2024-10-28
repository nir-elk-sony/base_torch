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
import random

import matplotlib.pyplot as plt
import timm

def add_soft_max(tensor_dict):
    tensor_dict['output-softmax'] = torch.nn.functional.softmax(torch.Tensor(tensor_dict['output']), dim=1)

def showim(I):
    m = I.min()
    M = I.max()
    plt.imshow(np.transpose((I-m)/(M-m), axes=(1,2,0)))
    plt.show()

# Get representative dataset generator
representative_dataset_gen = get_representative_dataset('C:/GIT/val_data_imagenet')


image = next(representative_dataset_gen())

timm_nets = ['tinynet_d.in1k','lcnet_075.ra2_in1k','mobilenetv3_small_075.lamb_in1k','lcnet_050.ra2_in1k',
             'mobilenetv3_small_100.lamb_in1k','lcnet_100.ra2_in1k','mobilenetv3_small_050.lamb_in1k',
             'tinynet_e.in1k','rexnet_100.nav_in1k','mobileone_s1.apple_in1k','tinynet_c.in1k',
             'tinynet_b.in1k','hardcorenas_a.miil_green_in1k','hardcorenas_b.miil_green_in1k',
             'efficientnet_b0.ra_in1k', 'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', 
             'efficientnet_b1.ra4_e3600_r240_in1k', 'mobilenetv2_100.ra_in1k']

if False:
    avail_pretrained_models = timm.list_models(pretrained=True)

    timm_nets = avail_pretrained_models

    random.shuffle(timm_nets)

    timm_nets = timm_nets[:1]

# not_working = ['rexnet_100.nav_in1k']
timm_nets = ['mobilenetv2_100.ra_in1k']
# timm_nets = list(set(timm_nets)-set(not_working))
# timm_nets = not_working

# model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# model = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
# timm_nets = ['beit_base_patch16_224.in22k_ft_in22k']
for net_name in timm_nets:
    model = timm.create_model(net_name, pretrained=True)
    model = model.eval()

    print(f"Test net: {net_name}")
    my_fx = my_Fx(model)
    if my_fx.fx_model is None:
        print(f"Test net: {net_name} -> FX error")
        continue
        
    

    tensor_dict = my_fx.forward(image)

    ref_out = model(image.clone().unsqueeze(0))
    assert (tensor_dict['output'] == ref_out).all().item()
    
    ref_out = my_fx.fx_model(image.clone().unsqueeze(0))
    assert (tensor_dict['output'] == ref_out).all().item()
    print(f"Test net: {net_name} -> OK")


raise KeyError()



import model_compression_toolkit as mct

net_name = 'efficientnet_b1.ra4_e3600_r240_in1k'
model = timm.create_model(net_name, pretrained=True)
# model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model = model.eval()

representative_dataset_gen = get_representative_dataset('C:/GIT/CIFAR10/val_data', n_iter=20, reshuffle_dir = True, reshuffle_gen = False, wrap_res_list=True, do_unsqueeze = True)

# Define a `TargetPlatformCapability` object, representing the HW specifications on which we wish to eventually deploy our quantized model.
target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap
)

mct.exporter.pytorch_export_model(quantized_model, save_model_path='qmodel.onnx', repr_dataset=representative_dataset_gen)
