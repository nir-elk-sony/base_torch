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
    