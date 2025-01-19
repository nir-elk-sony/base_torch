# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""

import torch
import torch.fx as fx
import json
import copy
from torch.nn.utils.fusion import fuse_conv_bn_eval



def fixed_res_node(v):
    
    if v.op == 'call_function' and v.target.__name__ == 'getattr' and len(v.args) == 2 and v.args[1] in ['shape', 'ndim']:
        return True
    return False
    
def replace_func_bn(model):
    try:
        fx_model= fx.symbolic_trace(model)
    except:
        fx_model= None
    if fx_model is None:
        return model

    graph = fx_model.graph        
    
    nodes = { node.name:node for node in graph.nodes}
    parameters = {k:m for k,m in fx_model.named_parameters() }
    buffers = {k:m for k,m in fx_model.named_buffers()}
            
    BN = [n for n in nodes.values() if n.op == 'call_function' and n.target == torch.nn.functional.batch_norm]
    for n in BN:
        
        mod = torch.nn.BatchNorm2d(num_features = buffers[n.args[1].target].shape[0], 
                             eps=n.kwargs.get('eps', 1e-05), 
                             momentum=n.kwargs.get('momentum', 0.1), 
                             affine='bias' in n.kwargs.keys(), 
                             track_running_stats=(len(n.args) > 1), 
                             device=None, dtype=None)

        
        assert not n.kwargs.get('training', False)            
        mod.running_mean = buffers[n.args[1].target]
        mod.running_var = buffers[n.args[2].target]
        mod.weight = parameters[n.kwargs['weight'].target]
        mod.bias   = parameters[n.kwargs['bias'].target]
        
        n.args = n.args[:1]
        
        n_kwargs = dict(n.kwargs)
        del n_kwargs['weight']
        del n_kwargs['bias']
        del n_kwargs['eps']
        del n_kwargs['momentum']
        del n_kwargs['training']
        
        n.kwargs = torch.fx.immutable_collections.immutable_dict(n_kwargs)

        n.target = n.name
        fx_model.add_submodule(n.name, mod)
        
        n.op = 'call_module'
    
    graph.lint()
    fx_model.recompile()
    return fx.GraphModule(fx_model, fx_model.graph)  


def _parent_name(target):
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


# Works for length 2 patterns with 2 modules
def matches_module_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
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


def replace_node_module(node, modules, new_module):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)

def fuse1(model: torch.nn.Module, inplace=False, no_trace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(torch.nn.Conv2d, torch.nn.BatchNorm2d)]
    
    if not inplace:
        model = copy.deepcopy(model)
    if not no_trace or not isinstance(model, torch.fx.GraphModule):
        fx_model = fx.symbolic_trace(model)
    else:
        fx_model = model
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                
                # Output of conv is used by other nodes
                if len([ n for n in node.args[0].users.keys() if not fixed_res_node(n)]) != 1:
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)





def fixed_data(t):
    if type(t) in [ tuple, list]:
        return all([fixed_data(tt) for tt in t])
    return type(t) in [int, str, bool, float, slice, type(None), torch.Size]

def op2exe(op, mods_fx):
    if op.op == 'call_function':
        return 'func:' + op.target.__name__
    if op.op == 'call_module':
        return type(mods_fx[op.target])
    if op.op == 'call_method':
        return 'method:' + op.target

class my_Fx:    
    def __init__(self, model):
        self.do_init_from_model(model)
                        
    def do_init_from_model(self, model):
        self.model = model
        
        try:
            self.fx_model= fx.symbolic_trace(model)
        except:
            self.fx_model= None
            
        if self.fx_model is None:
            return
        
        self.mods_fx = {k:m for k,m in self.fx_model.named_modules() }        
        
        # High-level intermediate representation (IR) - Graph representation
        self.graph = self.fx_model.graph        
        self.nodes = {node.name:node for node in self.graph.nodes}
        self.parameters = {k:m for k,m in self.fx_model.named_parameters() }
        self.buffers = {k:m for k,m in self.fx_model.named_buffers()}
                
        
        with open('fx_model.txt', 'w') as f:
            f.write(self.fx_model.code)
                        
        self.placeholders = [k for k,m in self.nodes.items() if m.op == 'placeholder']

        if False:        
            self.compute_input_output_dict()
                        
            #fixed_nodes = list({ k for k,v in self.nodes.items() if v.op == 'call_function' and v.target.__name__ == 'getattr' and len(v.args) == 2 and v.args[1] == 'shape'})
            fixed_nodes = list({ k for k,v in self.nodes.items() if fixed_res_node(v)})
            
            op_type = { k:v.op for k,v in self.nodes.items() }
            
            op_op = { k:str(op2exe(n, self.mods_fx)) for k,n in self.nodes.items() }
            
            with open(f'{type(self.model).__name__}.json', 'w') as f:
                json.dump({'op_type' : op_type, 
                           'nodes_outputs' : {k:list(v) for k,v in self.nodes_outputs.items() },
                           'fixed_nodes' : fixed_nodes,
                           'op_op': op_op}, f, indent=4)
                    

    def export_model_2_json(self, fname = None):
        nodes_outputs = self.compute_input_output_dict()
                    
        # fixed_nodes = list({ k for k,v in self.nodes.items() if v.op == 'call_function' and v.target.__name__ == 'getattr' and len(v.args) == 2 and v.args[1] == 'shape'})
        fixed_nodes = list({ k for k,v in self.nodes.items() if fixed_res_node(v)})
        op_type = { k:v.op for k,v in self.nodes.items() }
        
        op_op = { k:str(op2exe(n, self.mods_fx)) for k,n in self.nodes.items() }
        
        if fname is None:
            fname = f'{type(self.model).__name__}.json'
        with open(fname, 'w') as f:
            json.dump({'op_type' : op_type, 
                       'nodes_outputs' : {k:list(v) for k,v in nodes_outputs.items() },
                       'fixed_nodes' : fixed_nodes,
                       'op_op': op_op}, f, indent=4)
        



    def read_arg_by_ref(self, a, tensor_dict):

        if type(a) == torch.fx.node.Node:
            if a.op == 'get_attr':
                if a.target in self.buffers.keys():
                    arg = self.buffers[a.target]
                elif a.target in self.parameters.keys():
                    arg = self.parameters[a.target]
                else:
                    assert False, f'unknown input {a.target}'
            else:
                arg = tensor_dict.get(a.name)
            return arg
        
        if type(a) in [torch.fx.immutable_collections.immutable_list, tuple, list]:
            return [self.read_arg_by_ref(aa, tensor_dict) for aa in a]                
        if type(a) in [torch.fx.immutable_collections.immutable_dict, dict]:
            return {k:self.read_arg_by_ref(aa, tensor_dict) for k,aa in a.items() }
        assert fixed_data(a)
        return a

    def collect_tensors(self, L):
        r = set()
        if type(L) in [tuple, torch.fx.immutable_collections.immutable_list]:
            for in_node in L:
                r = r | self.collect_tensors(in_node)
            return r
        if type(L) in [torch.fx.immutable_collections.immutable_dict, dict]:
            for in_node in L.values():
                r = r | self.collect_tensors(in_node)
            return r
        
        if type(L) == torch.fx.node.Node:
            r.add(L.name)
            return r
        
        assert fixed_data(L)
        return r
        
    def compute_input_output_dict(self):
        nodes_inputs = dict()        
        for node in self.graph.nodes:
            nodes_inputs[node.name] = self.collect_tensors(node.args) | self.collect_tensors(node.kwargs)
        
        nodes_outputs = dict()
        for k,m in nodes_inputs.items():
            for r in m:
                nodes_outputs.setdefault(r, set())
                nodes_outputs[r].add(k)
        return nodes_outputs

    def forward(self, *inp):
        tensor_dict = dict()
        for k,m in zip(self.placeholders, inp):
            tensor_dict[k] = m
                    
        for op in self.graph.nodes:
                            
            if op.op in ['call_module', 'call_function', 'call_method']:
                    
                use_args = self.read_arg_by_ref(op.args, tensor_dict)
                kwargs = self.read_arg_by_ref(op.kwargs, tensor_dict)
                
                res = self.fx_apply(op,use_args, kwargs)
                                
                assert op.name not in tensor_dict.keys()

                if res is not None:
                    tensor_dict[op.name] = res

            elif op.op == 'output':
                assert len(op.args) == 1
                tensor_dict[op.name] = tensor_dict[op.args[0].name]
            elif op.op == 'placeholder':
                assert op.name in tensor_dict.keys()
            elif op.op == 'get_attr':                
                pass                
            else:
                assert False, f'unknown op: {op.op}'

        return tensor_dict

    def fx_apply(self, op, args, kwargs):

        if op.op == 'call_module':
            m = self.mods_fx[op.target]
        else:
            m = op.target
        if op.op == 'call_method':
            f = getattr(args[0],m)
            return f(*args[1:], **kwargs)        
        
        res = m(*args, **kwargs)
        return res
