# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch
import copy
from torch.fx import symbolic_trace
from imagenet_representative_dataset import get_representative_dataset
import numpy as np

def fixed_data(t):
    if type(t) in [ tuple, list]:
        return all([fixed_data(tt) for tt in t])
    return type(t) in [int, str, bool, float]
        
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

        self.parameters = {k:m for k,m in self.fx_model.named_parameters() }
        self.buffers = {k:m for k,m in self.fx_model.named_buffers()}
                
        with open('fx_model.txt', 'w') as f:
            f.write(self.fx_model.code)
            
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
                    if not fixed_data(in_node):
                        print(in_node, type(in_node))
                        assert False, 'unknown type'
        
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
                
        tensor_dict = dict()
        for k,m in zip(self.placeholders, inp):
            if type(m) == torch.Tensor:
                if len(m.shape) == 3:
                    m = m.unsqueeze(0).detach().numpy()
                else:
                    m = m.detach().numpy()
            tensor_dict[k] = m
                    
        # tensor_dict = { k:m.detach().numpy() for k,m in zip(self.placeholders, inp) }
        # tensor_dict = { k:m.clone() for k,m in zip(self.placeholders, inp) }
        
        for op_name in self.compute_order:
            # print("op_name:", op_name)
            op = self.nodes[op_name]
            if op.op in ['call_module', 'call_function', 'call_method']:
                if op.op == 'call_module':
                    m = self.mods_fx[op.target]
                else:
                    m = op.target
                use_args = []
                arg_names = []
                for a in op.args:
                    if type(a) == torch.fx.node.Node:
                        if a.op == 'get_attr':
                            if a.target in self.buffers.keys():
                                aa = self.buffers[a.target].clone()
                                arg_names.append(f'Buffer: {a.target}')
                            elif a.target in self.parameters.keys():
                                aa = self.parameters[a.target].clone()
                                arg_names.append(f'Parameter: {a.target}')
                            else:
                                assert False, f'unknown input {a.target}'
                        else:
                            aa = tensor_dict[a.name]
                            arg_names.append(f'Tensor: {a.name}')
                            if type(aa) == np.ndarray:
                                aa = torch.Tensor(aa.copy())
                            
                        use_args.append(aa)
                    else:
                        arg_names.append(f'Type: {a}')
                        use_args.append(a)

                res = self.fx_apply(m, op.op, *use_args)
                assert op.name not in tensor_dict.keys()
                if type(res) in [int, bool, type(None)]:
                    if res is not None:
                        tensor_dict[op.name] = res
                else:
                    if type(res) != torch.Tensor:
                        print("hh")
                    assert type(res) == torch.Tensor
                    tensor_dict[op.name] = res.detach().numpy()
                    
            elif op.op == 'output':
                assert len(op.args) == 1
                tensor_dict[op.name] = tensor_dict[op.args[0].name].copy()
            else:
                assert False, f'unknown op: {op.op}'

        return tensor_dict

    def get_layer_desc(self, compute_node, full=False):
        op = self.nodes[compute_node]
        if op.op == 'call_module':
            s = self.mods_fx[op.target]
            if not full:
                s = type(s).__name__
        elif op.op == 'call_function':
            s = op.target.__name__
        else:
            s = ''
        return s



    def fx_apply(self, m, op, *args):
        
        if op == 'call_method':
            f = getattr(args[0],m)
            if m == 'mean':
                kwargs = {'keepdim' : True}
            else:
                kwargs = dict()
            return f(*args[1:], **kwargs)
        
        if type(m) == torch.nn.modules.batchnorm.BatchNorm2d:
            if len(args[0].shape) == 3:
                return m(args[0].unsqueeze(0), *args[1:])
        if type(m).__name__ == 'function' and m.__name__ == 'batch_norm':
            if len(args[0].shape) == 3:
                return m(args[0].unsqueeze(0), *args[1:])            
        # print('fx_apply', m)
        res = m(*args)
        # if type(res) == torch.Tensor:
        #     print('res out shape', res.shape)
        return res


if __name__ == "__main__":
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
    