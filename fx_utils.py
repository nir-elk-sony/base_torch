# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
# from typing import Iterator, Tuple, List
# from torch.utils.data import DataLoader
import torch
import copy

from torchvision import transforms
# from torchvision.datasets import ImageFolder
import os
import random
from PIL import Image
import numpy as np
from torch.fx import symbolic_trace

        
    
class my_Fx:
    def __init__(self, model):
        self.model = model
        self.fx_model= symbolic_trace(model)
        self.mods_fx = {k:m for k,m in self.fx_model.named_modules() }        
        self.fx_model.eval()
        
        # High-level intermediate representation (IR) - Graph representation
        self.graph = copy.deepcopy(self.fx_model.graph)
        
        self.nodes = {node.name:node for node in self.graph.nodes}
        self.inputs = {k:m for k,m in self.nodes.items() if m.op == 'placeholder'}

        self.compute_input_output_dict()
        self.gen_compute_schdule()
        
    def compute_input_output_dict(self):
        self.nodes_inputs = dict()
        for node in self.graph.nodes:
            self.nodes_inputs[node.name] = set()
            for in_node in node.args:
                if type(in_node) == torch.fx.node.Node:
                    self.nodes_inputs[node.name].add(in_node.name)
                else:
                    print(in_node)
        
        self.nodes_outputs = dict()
        for k,m in self.nodes_inputs.items():
            for r in m:
                self.nodes_outputs.setdefault(r, set())
                self.nodes_outputs[r].add(k)

        self.placeholders = [k for k,m in self.nodes_inputs.items() if len(m) == 0]


    def gen_compute_schdule(self):
        nodes_inputs_invalids = copy.deepcopy(self.nodes_inputs)
        valid_calc_nodes = self.placeholders.copy()
        self.compute_order = [] 
        first = True
        while len(valid_calc_nodes) > 0:
            if not first:
                self.compute_order += valid_calc_nodes
            else:
                first = False
                
            new_valid_calc_nodes = []
            for p in valid_calc_nodes:
                for o in self.nodes_outputs.get(p, set()): 
                    nn = nodes_inputs_invalids[o]            
                    assert p in nn
                    nn.remove(p)
                    if len(nn) == 0:
                        new_valid_calc_nodes.append(o)
            valid_calc_nodes = new_valid_calc_nodes             

    def forward(self, *inp):
                
        tensor_dict = { k:m.clone() for k,m in zip(self.placeholders, inp) }
        
        for op_name in self.compute_order:
            op = self.nodes[op_name]
            if op.op in ['call_module', 'call_function']:
                if op.op == 'call_module':
                    m = self.mods_fx[op.target]
                else:
                    m = op.target
                use_args = []
                for a in op.args:
                    if type(a) == torch.fx.node.Node:
                        aa = tensor_dict[a.name].detach().numpy()
                        aa = torch.Tensor(aa).clone()                
                        use_args.append(aa)
                    else:
                        use_args.append(a)

                res = self.fx_apply(m, *use_args)
                assert op.name not in tensor_dict.keys()
                assert type(res) == torch.Tensor
                tensor_dict[op.name] = res.clone()
            elif op.op == 'output':
                assert len(op.args) == 1
                tensor_dict[op.name] = tensor_dict[op.args[0].name].clone()

        return tensor_dict

    def fx_apply(self, m, *args):
        if type(m) == torch.nn.modules.batchnorm.BatchNorm2d:
            if len(args[0].shape) == 3:
                return m(args[0].unsqueeze(0), *args[1:])
        return m(*args)



model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()

# Define transformations for the validation set
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

src_dir = 'C:/GIT/CIFAR10/val_data'
L = os.listdir(src_dir)

# Define representative dataset generator
def get_representative_dataset(src_dir, image_file_list, val_transform):

    def representative_dataset():
        file_order = image_file_list.copy()
        random.shuffle(file_order)
        for fname in file_order:
            image = Image.open(f'{src_dir}/{fname}').convert('RGB')
            image = val_transform(image)
            yield image.clone()
            
    return representative_dataset

# Get representative dataset generator
representative_dataset_gen = get_representative_dataset(src_dir, L, val_transform)

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

# m=my_fx.mods_fx['features.8.conv.0.0']





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



for _ in range(100):
    n0 = np.random.randn(1000)-3
    n  = np.random.randn(1000)*0.01
    
    n1 = n0.copy()
    n1[n0 < 0] = 0
    
    n2 = (n0+n).copy()
    n2[n2 < 0] = 0
    
    n1 = n1-n1.mean()
    n2 = n2-n2.mean()
    
    print(np.sqrt(((n1)*(n1)).mean())/np.sqrt(((n1-n2)*(n1-n2)).mean()))
