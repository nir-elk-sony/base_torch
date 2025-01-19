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
from fx_wrapper import my_Fx, replace_func_bn, fuse1
import random
from torch.fx.experimental.optimization import fuse, remove_dropout

import matplotlib.pyplot as plt
import timm

from quantized_wrapper import RecoderWrapper

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# def add_soft_max(tensor_dict):
#     tensor_dict['output-softmax'] = torch.nn.functional.softmax(torch.Tensor(tensor_dict['output']), dim=1)

# def showim(I):
#     m = I.min()
#     M = I.max()
#     plt.imshow(np.transpose((I-m)/(M-m), axes=(1,2,0)))
#     plt.show()

# Get representative dataset generator
representative_dataset_gen = get_representative_dataset('C:/GIT/val_data_imagenet')

image = next(representative_dataset_gen())

image = image.unsqueeze(0)

timm_nets = ['tinynet_d.in1k','lcnet_075.ra2_in1k','mobilenetv3_small_075.lamb_in1k','lcnet_050.ra2_in1k',
             'mobilenetv3_small_100.lamb_in1k','lcnet_100.ra2_in1k','mobilenetv3_small_050.lamb_in1k',
             'tinynet_e.in1k','rexnet_100.nav_in1k','mobileone_s1.apple_in1k','tinynet_c.in1k',
             'tinynet_b.in1k','hardcorenas_a.miil_green_in1k','hardcorenas_b.miil_green_in1k',
             'efficientnet_b0.ra_in1k', 'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', 
             'efficientnet_b1.ra4_e3600_r240_in1k', 'mobilenetv2_100.ra_in1k']

if True:
    avail_pretrained_models = timm.list_models(pretrained=True)

    timm_nets = avail_pretrained_models

    random.shuffle(timm_nets)

    # timm_nets = timm_nets[:1]

not_working = ['rexnet_100.nav_in1k']
not_working = ['beit_base_patch16_384.in22k_ft_in22k_in1k']
# timm_nets = list(set(timm_nets)-set(not_working))
# timm_nets = not_working

# model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# model = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
timm_nets = ['beit_base_patch16_224.in22k_ft_in22k']

choices=['vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'deit_base', 'swin_tiny', 'swin_small']
timm_nets = ['vit_tiny_patch16_224.augreg_in21k']
timm_nets = ['mobilenetv2_100.ra_in1k']
timm_nets = ['mobilenetv3_small_100.lamb_in1k']

timm_nets = ['resnet50.b2k_in1k']



timm_nets = ['vit_tiny_patch16_224.augreg_in21k', 'mobilenetv2_100.ra_in1k', 'mobilenetv3_small_100.lamb_in1k', 'resnet50.b2k_in1k']
timm_nets = ['mobilenetv3_small_100.lamb_in1k']
# timm_nets = ['mobilenetv2_100.ra_in1k']

for net_name in timm_nets:
    if net_name in not_working:
        continue
    model = timm.create_model(net_name, pretrained=True)    
    model = model.eval()
    
    fname = f'{type(model).__name__}.json'
    
    ref_out = model(image)
    if True:
        model = remove_dropout(model)
        model = replace_func_bn(model)
        model = model.eval()
        model = fuse1(model)
        model = model.eval()
    
    # Validate substitution didn't make much change!
    ref_out1 = model(image)
    assert torch.isclose(ref_out1, ref_out, rtol=1e-4, atol=1e-4).all().item()
    assert torch.abs(ref_out1/ref_out-1.0).max() < 1e-2*torch.abs(ref_out).max()
    
    print(f"Test net: {net_name}")
    my_fx = my_Fx(model)
    my_fx.export_model_2_json(fname)
    
    if my_fx.fx_model is None:
        print(f"Test net: {net_name} -> FX error")
        continue
        
    if False:
        nodes_names, _ = get_graph_node_names(model)        
        feature_extractor = create_feature_extractor(model, return_nodes=nodes_names)
        # `out` will be a dict of Tensors, each representing a feature map
        out = feature_extractor(image.clone().unsqueeze(0))
    else:
        out = None
    
    
    tensor_dict = my_fx.forward(image)
    
    if False:
        for k in out.keys() & tensor_dict.keys():
            if type(out[k]) == tuple:
                for a,b in zip(out[k], tensor_dict[k]):
                    eq = (a == b)
                    # print(k, eq.all())
                    if type(eq) == torch.Tensor:
                        print(k, eq.all())
            else:
                eq = (out[k] == tensor_dict[k])
                # print(k, eq.all())
                if type(eq) == torch.Tensor:
                    print(k, eq.all())
        
    with torch.no_grad():
        ref_out = model(image)
    assert (tensor_dict['output'] == ref_out).all().item()

    with torch.no_grad():
        ref_out1 = my_fx.fx_model(image)
    assert (tensor_dict['output'] == ref_out1).all().item()
    print(f"Test net: {net_name} -> OK")

if False:
    
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
