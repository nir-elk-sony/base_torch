# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:40:29 2024

@author: 7000028246
"""

from torch import fx,nn
import torch
import copy
from quantized_wrapper import QuantizedWrapper, ActivationQuantizedWrapper
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from fx_utils import my_Fx

## New code - replace q

LINEAR_OPS_list = [nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear]
ACTIVATION_OPS_list = [nn.ReLU,nn.ReLU6,nn.Identity]

LINEAR_OPS = [(nn.Conv1d,),
              (nn.Conv2d,),
              (nn.Conv3d,),
              (nn.Linear,)]

ACTIVATION_OPS = [(nn.ReLU,),
                  (nn.ReLU6,),
                  (nn.Identity,)]


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

def replace2quantized_model(in_model, skip_layers = [], linear_patterns=LINEAR_OPS, 
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
                    # if in_node.target in skip_layers:
                        # continue
                    
                    _matches_module_pattern(pattern, node, in_node, modules)
                    
                    
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
        
        last_linear_layer = [n for n, m in model_fold.named_modules() if isinstance(m, tuple([t[0] for t in LINEAR_OPS]))][-1]
        
        self.qmodel = replace2quantized_model(model_fold, skip_layers = [last_linear_layer])

        self.qmodel.train(False)

        print("End Layer Wrapping")

    def __call__(self, x):
        return self.qmodel(x)

    def apply_weights_quantization(self):
        print("Apply Quantization")
        for n, m in self.qmodel.named_modules():
            if isinstance(m, QuantizedWrapper):
                m.apply_weights_quantization()


model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fx_model = fx.symbolic_trace(model)
modules = dict(fx_model.named_modules())

c = list(fx_model.children())
c = c[1]
dict(c.named_children())

c = dict(fx_model.named_children())

d = {n:m for n,m in fx_model.named_modules() if type(m) == nn.Conv2d}


nodes = list(fx_model.graph.nodes)


my_fx = my_Fx(fuse(model))

my_fx.nodes_inputs

pattern = [LINEAR_OPS_list, ACTIVATION_OPS_list]


for rest_names, rest_m in my_fx.iter_pattern(pattern):
    if len(rest_names) == 2:
        my_fx.wrap_module(rest_names[0], QuantizedWrapper(rest_m[0], name=rest_names[0]))
        my_fx.wrap_module(rest_names[1], ActivationQuantizedWrapper(rest_m[1], name=rest_names[1]))
    else:
        new_w_q = QuantizedWrapper(rest_m[0], name=rest_names[0])
        new_a_q = ActivationQuantizedWrapper(new_w_q, name=rest_names[0])
        my_fx.wrap_module(rest_names[0], new_a_q)
        
my_fx.update_changes()        
        
# qmodel = fx.GraphModule(my_fx.fx_model, my_fx.fx_model.graph)
# my_fx = my_Fx(qmodel)


for name,node in my_fx.iter_nodes(ACTIVATION_OPS_list):
    if not name.endswith('act_op'):
        my_fx.wrap_module(name, ActivationQuantizedWrapper(node, name=name))
        print(name)

for name,node in my_fx.iter_nodes(LINEAR_OPS_list):
    if not name.endswith('weight_op'):
        my_fx.wrap_module(name, QuantizedWrapper(node, name=name))
        print(name)

my_fx.update_changes()

# qmodel = fx.GraphModule(my_fx.fx_model, my_fx.fx_model.graph)
# my_fx = my_Fx(qmodel)


for name,node in my_fx.iter_nodes(ACTIVATION_OPS_list):
    if not name.endswith('act_op'):
        print(name)

for name,node in my_fx.iter_nodes(LINEAR_OPS_list):
    if not name.endswith('weight_op'):
        print(name)

qmodel = my_fx.get_updated_model()

# for rest_names, rest_m in my_fx.iter_pattern(pattern):
#     print(rest_names, rest_m)

# for name,node in my_fx.nodes.items():


#     rest_names, rest_m  = my_fx.matched_pattern(name, node, pattern)
#     if rest_names is not None:
#         my_fx.wrap_module(rest_names[0], QuantizedWrapper(rest_m[0], name=rest_names[0]))                
#         my_fx.wrap_module(rest_names[1], ActivationQuantizedWrapper(rest_m[1], name=rest_names[1]))


print(qmodel.__repr__)

raise KeyError()
    
    
    # m = my_fx.get_node_operation(node)
    # mod_list = [m]
    # mod_names = [name]
    # if type(mod_list[-1]) in pattern[0]:
    #     next_nodes = my_fx.get_output_nodes(name)
    #     if len(next_nodes) == 1: 
    #         a = my_fx.get_node_operation(next_nodes[0])
    #         if type(a) in pattern[1]:
    #             mod_list.append(a)
    #             mod_names.append(next_nodes[0].name)
                
                
    #             my_fx.wrap_module(mod_names[0], QuantizedWrapper(mod_list[0], name=mod_names[0]))                
    #             my_fx.wrap_module(mod_names[1], ActivationQuantizedWrapper(mod_list[1], name=mod_names[1]))
                
    #             mod_list.pop()
    #             mod_names.pop()
                
            
            # if type(a) in pattern[1]:

            #     my_fx.wrap_module(name, QuantizedWrapper(m, name=name))                
            #     my_fx.wrap_module(next_nodes[0].name, ActivationQuantizedWrapper(a, name=name))                
                
                
                
                # print('*'*50)
                # print(name, next_nodes[0].name)
                            
                # _,pp = my_fx.get_module_and_parent_by_name(name)
                # # assert mm == m
                # # print('*'*50)
                # # print(mm)
                # # print(pp)
    
                # new_module = QuantizedWrapper(m, name=name)
                # child_name = [n for n,mmm in pp.named_children() if mmm == m ]
                # assert len(child_name) == 1
                # setattr(pp, child_name[0], new_module)
                
                # # wrap_node = _replace_node_module(in_node, modules, QuantizedWrapper(mm, name=name))
    
                # # assert(isinstance(node.target, str))
                # # parent_name, name = _parent_name(node.target)
                # # modules[node.target] = new_module
                # # setattr(modules[parent_name], name, new_module)
                # # return new_module
    
    
    
    
    
                
                
                
                
                # _,pp = my_fx.get_module_and_parent_by_name(next_nodes[0].name)
                # # assert mm == a
                # # print('*'*50)
                # # print(mm)
                # # print(pp)
                
                # new_module = ActivationQuantizedWrapper(a, name=name)
                # child_name = [n for n,mmm in pp.named_children() if mmm == a ]
                # assert len(child_name) == 1
                # setattr(pp, child_name[0], new_module)
                
                # # name1 = my_fx.graph_names_2_module_names[name]
                # # print(my_fx.module_by_calc[name1])
                # # print(my_fx.parent_by_name[name1])
                # # print(my_fx.module_by_calc[my_fx.parent_by_name[name1]])
    
                # # name1 = my_fx.graph_names_2_module_names[next_nodes[0].name]
                
                # # print(my_fx.module_by_calc[name1])
                # # print(my_fx.parent_by_name[name1])
                # # print(my_fx.module_by_calc[my_fx.parent_by_name[name1]])


qmodel = fx.GraphModule(my_fx.fx_model, my_fx.fx_model.graph)


raise KeyError()



# raise KeyError()

model = model.eval()


qm = QuantizationModelWrapper(model)



