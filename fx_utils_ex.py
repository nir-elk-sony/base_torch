# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:40:29 2024

@author: 7000028246
"""

from torch import nn
from quantized_wrapper import QuantizedWrapper, ActivationQuantizedWrapper
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from fx_utils import my_Fx
from torch.fx.experimental.optimization import fuse

LINEAR_OPS_list = [nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear]
ACTIVATION_OPS_list = [nn.ReLU,nn.ReLU6,nn.Identity]
pattern = [LINEAR_OPS_list, ACTIVATION_OPS_list]

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()

my_fx = my_Fx(fuse(model))


for rest_names, rest_m in my_fx.iter_pattern(pattern):
    if len(rest_names) == 2:
        my_fx.wrap_module(rest_names[0], QuantizedWrapper(rest_m[0], name=rest_names[0]))
        my_fx.wrap_module(rest_names[1], ActivationQuantizedWrapper(rest_m[1], name=rest_names[1]))
    else:
        new_w_q = QuantizedWrapper(rest_m[0], name=rest_names[0])
        new_a_q = ActivationQuantizedWrapper(new_w_q, name=rest_names[0])
        my_fx.wrap_module(rest_names[0], new_a_q)
        
my_fx.update_changes()        

for name,node in my_fx.iter_nodes(ACTIVATION_OPS_list):
    if not name.endswith('act_op'):
        my_fx.wrap_module(name, ActivationQuantizedWrapper(node, name=name))
        print(name)

for name,node in my_fx.iter_nodes(LINEAR_OPS_list):
    if not name.endswith('weight_op'):
        my_fx.wrap_module(name, QuantizedWrapper(node, name=name))
        print(name)

my_fx.update_changes()


for name,node in my_fx.iter_nodes(ACTIVATION_OPS_list):
    if not name.endswith('act_op'):
        print(name)

for name,node in my_fx.iter_nodes(LINEAR_OPS_list):
    if not name.endswith('weight_op'):
        print(name)

qmodel = my_fx.get_updated_model()


print(qmodel.__repr__)
