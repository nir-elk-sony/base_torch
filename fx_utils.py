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
    return type(t) in [int, str, bool, float, slice]
        
class my_Fx:
    def __init__(self, model):
        self.model = model
        self.fx_model= symbolic_trace(model)
        self.mods_fx = {k:m for k,m in self.fx_model.named_modules() }        
        
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




    def read_arg_by_ref(self, a, tensor_dict):

        arg_name = None
        arg = None
        if type(a) == torch.fx.node.Node:
            if a.op == 'get_attr':
                if a.target in self.buffers.keys():
                    aa = self.buffers[a.target].clone()
                    arg_name = f'Buffer: {a.target}'
                elif a.target in self.parameters.keys():
                    aa = self.parameters[a.target].clone()
                    arg_name = f'Parameter: {a.target}'
                else:
                    assert False, f'unknown input {a.target}'
            else:
                aa = tensor_dict[a.name]
                arg_name = f'Tensor: {a.name}'
                if type(aa) == np.ndarray:
                    aa = torch.Tensor(aa.copy())
                
            arg = aa
        elif type(a) == torch.fx.immutable_collections.immutable_list:
            arg_name = []
            arg = []
            for aa in a:
                arg1, arg_name1 = self.read_arg_by_ref(aa, tensor_dict)
                arg_name.append(arg_name1)
                arg.append(arg1)
            pass
        else:
            arg_name = f'Type: {a}'
            arg = a
            
        return arg, arg_name




    def collect_tensors(self, L):
        r = set()        
        for in_node in L:
            if type(in_node) == torch.fx.node.Node:
                r.add(in_node.name)
            elif type(in_node) == torch.fx.immutable_collections.immutable_list:
                r = r | self.collect_tensors(in_node)
            else:
                if not fixed_data(in_node):
                    print(in_node, type(in_node))
                    assert False, 'unknown type'
        return r

        
    def compute_input_output_dict(self):
        self.nodes_inputs = dict()
        for node in self.graph.nodes:
            self.nodes_inputs[node.name] = self.collect_tensors(node.args)
        
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
                    arg, arg_name = self.read_arg_by_ref(a, tensor_dict)
                    use_args.append(arg)
                    arg_names.append(arg_name)
                    
                kwargs = dict()
                for k,a in op.kwargs.items():
                    arg, arg_name = self.read_arg_by_ref(a, tensor_dict)
                    kwargs[k] = arg
                    

                res = self.fx_apply(m, op.op, use_args, kwargs)
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



    def fx_apply(self, m, op, args, kwargs):
        
        if op == 'call_method':
            f = getattr(args[0],m)
            # if m == 'mean':
                # kwargs = {'keepdim' : True}
            # else:
                # kwargs = dict()
            return f(*args[1:], **kwargs)
        
        if type(m) == torch.nn.modules.batchnorm.BatchNorm2d:
            if len(args[0].shape) == 3:
                return m(args[0].unsqueeze(0), *args[1:])
        if type(m).__name__ == 'function' and m.__name__ == 'batch_norm':
            if len(args[0].shape) == 3:
                return m(args[0].unsqueeze(0), *args[1:])            
        # print('fx_apply', m)
        res = m(*args, **kwargs)
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
    
    
    
    
    




def find_tensor_in_dict(t, d):
    assert type(t) == torch.Tensor
    for k,tt in d.items():
        if type(tt) in [torch.Tensor, np.ndarray]:
            if t.shape == tt.shape:
                if (t == tt).all().item():
                    return k

def pre_hook(model, inp, T):
    
    # if type(model) in [torch.nn.modules.container.Sequential, torch.nn.modules.dropout.Dropout]:
        # return
    # if type(model) in [torch.nn.modules.container.Sequential]:
        # return
    for ii in inp:
        print(find_tensor_in_dict(ii, T))
    return
    kk = None
    for k,m in mods.items():
        if model == m:
            kk = k
    if kk is None:
        assert False
        return

    inp_tensors = []
    if type(inp) == tuple:
        for ii in inp:
            if type(ii) == torch.fx.proxy.Proxy:
                return                
            assert type(ii) == torch.Tensor
            inp_name = find_tensor_in_dict(ii, tensors_dict)
            if inp_name is None:
                print("hhh")
            inp_tensors.append(inp_name)
    else:
        if type(inp) == torch.fx.proxy.Proxy:
            return                
        assert type(inp) == torch.Tensor
        inp_tensors.append(find_tensor_in_dict(inp, tensors_dict))
    
    op_list.append((kk, inp_tensors))

def hook(model, inp, out, T):

    # if type(model) in [torch.nn.modules.container.Sequential]:
        # return
    # if hasattr(model, 'inplace'):
        # print(out.shape)
        # print("in place")
    # if type(model) in [torch.nn.modules.container.Sequential, torch.nn.modules.dropout.Dropout]:
    #     return
    print(find_tensor_in_dict(out, T))
    
    return
    kk = None
    for k,m in mods.items():
        if model == m:
            kk = k
    if kk is None:
        assert False
        return

    if type(out) == tuple:
        for ix,ii in enumerate(out):
            if type(ii) == torch.fx.proxy.Proxy:
                return                
            assert type(ii) == torch.Tensor
            tensors_dict[(kk,ix)] = ii.clone()
    else:
        if type(out) == torch.fx.proxy.Proxy:
            return                
        assert type(out) == torch.Tensor
        tensors_dict[(kk,0)] = out.clone()
            
    