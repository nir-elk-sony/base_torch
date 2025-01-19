# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch
import copy
# from torch.fx import symbolic_trace, GraphModule
import torch.fx as fx

from imagenet_representative_dataset import get_representative_dataset
import numpy as np

def fixed_data(t):
    assert False
    if type(t) in [ tuple, list]:
        return all([fixed_data(tt) for tt in t])
    return type(t) in [int, str, bool, float, slice, type(None), torch.Size]
        
# Todo: gen pointer from module to the parent module
def op2exe(op, mods_fx):
    assert False
    if op.op == 'call_function':
        return 'func:' + op.target.__name__
    if op.op == 'call_module':
        return type(mods_fx[op.target])
    if op.op == 'call_method':
        return 'method:' + op.target
    


class my_Fx:
    
    def __init__(self, model):
        self.do_init_from_model(model)


    def op2str(self, op):
        assert False
        if op.op == 'placeholder':
            return f'Placeholder\t\t: {op.target}'
        if op.op == 'call_function':
            return f'Call function\t: {op.name} = {op.target.__name__}(...)'
        if op.op == 'call_module':
            return f'Call module\t\t: {op.name} = {str(self.mods_fx[op.target])}'
        if op.op == 'call_method':
            return f'Call method\t\t: {op.name} = {op.args[0].name}.{op.target}(...)'
        if op.op == 'get_attr':
            if op.target in self.parameters.keys():
                return f'Parameter\t\t: {op.name} = self.{op.target}'
            if op.target in self.buffers.keys():
                return f'Buffer\t\t\t: {op.name} = self.{op.target}'
            assert False, f'unknown attr: {op.name} = self.{op.target}'
        if op.op == 'output':
            return f'Output\t\t: {op.name} = {op.args[0].name}.{op.target}(...)'
            
        print(op, op.op, op.target)
        assert False
        

    def get_updated_model(self):
        return fx.GraphModule(self.fx_model, self.fx_model.graph)  
        
    def update_changes(self):        
        self.do_init_from_model(self.get_updated_model())
                
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
        # self.graph = copy.deepcopy(self.fx_model.graph)
        self.graph = self.fx_model.graph
        
        self.nodes = {node.name:node for node in self.graph.nodes}
        self.inputs = {k:m for k,m in self.nodes.items() if m.op == 'placeholder'}

        self.parameters = {k:m for k,m in self.fx_model.named_parameters() }
        self.buffers = {k:m for k,m in self.fx_model.named_buffers()}
                
        with open('fx_model.txt', 'w') as f:
            f.write(self.fx_model.code)
            
        
        if False:
            dd = {k:m for k,m in self.fx_model.named_modules() }
            for k,m in dd.items():
                print('*'*40)
                print(k)
                print('*'*40)
                print(m)
                print()
                print()
            
            
        self.placeholders = {k for k,m in self.nodes.items() if m.op == 'placeholder'}

        #self.compute_input_output_dict()
        #self.gen_compute_schdule()
        
        
        self.calc_module = {n:m for n,m in self.fx_model.named_modules() if type(m) != torch.nn.modules.module.Module}
        
        self.module_by_calc = dict()
        self.parent_by_name = dict()
        
        self.update_module_by_calc_dict(self.fx_model, '')
        # print(self.module_by_calc)
        
        #self.calc_dungling_nodes()
        
        if False:
            
            #op_type = { k:v.op for k,v in self.nodes.items() if v.op == 'get_attr'}
            fixed_nodes = list({ k for k,v in self.nodes.items() if v.op == 'call_function' and v.target.__name__ == 'getattr' and len(v.args) == 2 and v.args[1] == 'shape'})
            
            
            
            import json
            op_type = { k:v.op for k,v in self.nodes.items() }
            
            op_op = { k:str(op2exe(n, self.mods_fx)) for k,n in self.nodes.items() }
            
            with open(f'{type(self.model).__name__}.json', 'w') as f:
                json.dump({'op_type' : op_type, 
                           'nodes_outputs' : {k:list(v) for k,v in self.nodes_outputs.items() },
                           'fixed_nodes' : fixed_nodes,
                           'op_op': op_op}, f, indent=4)
            
            
            
            import pydot
            
            
            node_colors = {'output' : 'red', 'get_attr' : 'yellow', 'call_method' : 'gray', 'call_function' : 'coral', 'call_module' : 'azure', 'placeholder' : 'blue'}
            
            all_nodes_by_name = set(sum([list(x) for x in self.nodes_outputs.values() ], []))
            
        
            # graph = pydot.Dot("my_graph", graph_type="graph") # , bgcolor="yellow")
            graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor="white")
                                                                                           
            # Add nodes
            for n in all_nodes_by_name:
                
                graph.add_node(pydot.Node(n, label=n, bgcolor=node_colors.get(n, 'white')))

            for s,d in self.nodes_outputs.items():
                # Add edges
                for dd in d:
                    graph.add_edge(pydot.Edge(src=s, dst=dd))
            # graph.add_edge(pydot.Edge(src="b", dst="c", color="red"))
            
            graph.write_png("output.png")
        
        
        
    def update_module_by_calc_dict(self, mod, name):
        
        # print('*'*50)
        # print(mod.__repr__)
        # print(name)
        # print('*'*50)
        self.module_by_calc[name] = mod
        for n,m in mod.named_children():
            next_name = f'{name}.{n}' if len(name) > 0 else n
            self.parent_by_name[next_name] = name
            if hasattr(m, 'named_children'):
                self.update_module_by_calc_dict(m,next_name)
            else:
                self.module_by_calc[next_name] = m
        
        
        

    def get_module_and_parent_by_name(self, name):
        name1 = self.graph_names_2_module_names[name]
        m = self.module_by_calc[name1]
        p = self.module_by_calc[self.parent_by_name[name1]]
        return m,p
    







    def wrap_module(self, name, new_module):                
                                            
        mm,pp = self.get_module_and_parent_by_name(name)    
        child_name = [n for n,mmm in pp.named_children() if mmm == mm ]
        assert len(child_name) == 1
        setattr(pp, child_name[0], new_module)


    def matched_pattern(self, node, pattern):
        assert len(pattern) > 0
        next_nodes = self.get_output_nodes(node.name)
        m = self.get_node_operation(node)
                
        if type(m) in pattern[0] and len(next_nodes) == 1:
            
            if len(pattern) == 1:
                return [node.name],[m]
            else:
                res = self.matched_pattern(next_nodes[0], pattern[1:])
                if res is not None:
                    return [node.name]+res[0], [m]+res[1]
                else:
                    return [node.name],[m]
                  
    def iter_all_nodes(self):
        for name,node in self.nodes.items():
            m = self.get_node_operation(node)
            yield name,m
        
    def iter_nodes(self, node_types):
        for name,node in self.nodes.items():
            m = self.get_node_operation(node)
            if type(m) in node_types:
                yield name,m
                
    def iter_pattern(self, pattern):
        
        for name,node in self.nodes.items():
            res = self.matched_pattern(node, pattern)
            if res is not None:
                yield res


    def read_output_by_ref(self, a):
        return a
    
        if fixed_data(a):
            return a
        if type(a) in [tuple, list]:
            return tuple(self.read_output_by_ref(aa) for aa in a)
        if type(a) == np.ndarray:
            return a.copy()
        if type(a) != torch.Tensor:
            assert False, f'unexpected output type: {type(a)}'
        assert type(a) == torch.Tensor
        return a.detach().numpy()


    def read_arg_by_ref(self, a, tensor_dict, reference):

        arg_name = None
        arg = None
        if type(a) == torch.fx.node.Node:
            if a.op == 'get_attr':
                if a.target in self.buffers.keys():
                    aa = self.buffers[a.target] # .clone()
                    arg_name = f'Buffer: {a.target}'
                elif a.target in self.parameters.keys():
                    aa = self.parameters[a.target] # .clone()
                    arg_name = f'Parameter: {a.target}'
                else:
                    assert False, f'unknown input {a.target}'
            else:
                aa = tensor_dict[a.name]
                arg_name = f'Tensor: {a.name}'
                # r = reference[a.name]
                
                # for k,m in reference.items():
                #     if type(m) == type(aa):
                #         print(k, m == aa)
                if False:
                    if type(aa) == np.ndarray:
                        # add this since in some calculation the operation with grad produce different result that without grad
                        if False:
                            aa = torch.Tensor(aa.copy()).requires_grad_()
                        else:
                            aa = torch.Tensor(aa.copy())
                
            arg = aa
        elif type(a) in [torch.fx.immutable_collections.immutable_list, tuple, list]:
            arg_name = []
            arg = []
            for aa in a:
                arg1, arg_name1 = self.read_arg_by_ref(aa, tensor_dict, reference)
                arg_name.append(arg_name1)
                arg.append(arg1)
                
        elif type(a) in [torch.fx.immutable_collections.immutable_dict, dict]:

            arg_name = dict()
            arg = dict()
            for k,aa in a.items():
                arg[k], arg_name[k] = self.read_arg_by_ref(aa, tensor_dict, reference)
        else:
            arg_name = f'Type: {a}'
            if not fixed_data(a):
                print(a, type(a))
            assert fixed_data(a)
            arg = a
            
        return arg, arg_name

    
    def calc_dungling_nodes(self):
        
        nodes_outputs = self.nodes_outputs.copy()
        nodes_inputs = self.nodes_inputs.copy()
        
        nodes = list(self.graph.nodes)
        nodes_dict = { n.name:n for n in self.graph.nodes}
        
        
        
        dungling_nodes = [ n for n in nodes if n.name not in nodes_outputs.keys() and n.op != 'output' ] + [nodes_dict[k] for k,v in nodes_outputs.items() if len(v) == 0]
        dungling_nodes = list(set(dungling_nodes))
        dungling_nodes_set = set([d.name for d in dungling_nodes])
        
        for nn in dungling_nodes: 
            inps = [ nodes_dict[nn] for nn in nodes_inputs[nn.name] ]
            for inps1 in inps:
                print(nodes_outputs[inps1.name] & dungling_nodes_set)
                print()
                nodes_outputs[inps1.name] = nodes_outputs[inps1.name]-dungling_nodes_set


    def collect_tensors(self, L):
        r = set()        
        for in_node in L:
            if type(in_node) == torch.fx.node.Node:
                r.add(in_node.name)
            elif type(in_node) in [tuple, torch.fx.immutable_collections.immutable_list]:
                r = r | self.collect_tensors(in_node)
            else:
                if not fixed_data(in_node):
                    print(in_node, type(in_node))
                    assert False, 'unknown type'
        return r

        
    def compute_input_output_dict(self):
        self.nodes_inputs = dict()
        self.graph_names_2_module_names = dict()
        
        for node in self.graph.nodes:
            self.nodes_inputs[node.name] = self.collect_tensors(node.args)
            if node.op == 'call_module':
                self.graph_names_2_module_names[node.name] = node.target
                
            
        
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


    def get_node_operation(self, node):
        if node.op == 'call_module':        
            return self.mods_fx.get(node.target)
        
        if node.op == 'call_function':
            return node.target



    def forward(self, *inp, **kwargs):
                
        reference = kwargs.get('reference')
        tensor_dict = dict()
        for k,m in zip(self.placeholders, inp):
            if type(m) == torch.Tensor:
                if len(m.shape) == 3:
                    m = m.unsqueeze(0)
                
                # m = m.detach().numpy()
            tensor_dict[k] = m
                    
        # tensor_dict = { k:m.detach().numpy() for k,m in zip(self.placeholders, inp) }
        # tensor_dict = { k:m.clone() for k,m in zip(self.placeholders, inp) }
        self.calc_op_list = []
        self.calc_op_list1 = []




        # for op in self.graph.nodes:
        #     op_name = op.name
            
        #     print(self.op2str(op))
        
        
        # # for op_name in self.compute_order:
        # #     # print("op_name:", op_name)
        # #     # print("")
        # #     op = self.nodes[op_name]
            
        #     if op.op in ['call_module', 'call_function', 'call_method']:
        #         if op.op == 'call_module':
        #             m = self.mods_fx[op.target]
        #         else:
        #             m = op.target
        
        
        if False:
            nodes = list(self.graph.nodes)
            ops = [op2exe(op1, self.mods_fx) for op1 in self.graph.nodes]
            op_set = set(ops)
            for ix, op in enumerate(op_set):
                print('*'*50)
                print(ix, op)

        for op in self.graph.nodes:
            op_name = op.name
                
                #op_str = self.op2str(op)
                #self.calc_op_list1.append(op_str)
            # print(op_str)
        
        
        # for op_name in self.compute_order:
        #     # print("op_name:", op_name)
        #     # print("")
        #     op = self.nodes[op_name]
            
            if op.op in ['call_module', 'call_function', 'call_method']:
                if op.op == 'call_module':
                    m = self.mods_fx[op.target]
                else:
                    m = op.target
                    
                use_args, arg_names = self.read_arg_by_ref(op.args, tensor_dict, reference)
                # if len(op.kwargs):
                    # print("hhh")
                kwargs, kwargs_name = self.read_arg_by_ref(op.kwargs, tensor_dict, reference)

                self.calc_op_list.append((op_name, str(m), op.kwargs, op.args))
                # print(op_name)
                res = self.fx_apply(m, op.op, use_args, kwargs)
                
                if False:
                    ref = reference.get(op.name)
                    if ref is not None:
                        if type(res) == torch.Tensor:
                            b = (reference.get(op.name) == res).all()
                            print(op.name, b)
                            if not b:
                                print('hh')
                
                assert op.name not in tensor_dict.keys()

                res = self.read_output_by_ref(res)
                if res is not None:
                    tensor_dict[op.name] = res

            elif op.op == 'output':
                assert len(op.args) == 1
                tensor_dict[op.name] = tensor_dict[op.args[0].name]
            elif op.op == 'placeholder':
                # print(op)
                assert op.name in tensor_dict.keys()
            elif op.op == 'get_attr':                
                pass
                # print(op)
                
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

    def get_output_nodes(self, node_name):
        
        return [ self.nodes[mm] for mm in self.nodes_outputs.get(node_name, []) ]

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
            
    