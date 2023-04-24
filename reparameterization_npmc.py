import argparse
from typing import *
import copy
import yaml

import torch
import torch.nn as nn
import torch.fx as fx

implicit_layer_number = {
    'yolov7-tiny': 77,
    'yolov7': 105,
    'yolov7x': 121,
    'yolov7-w6': 122,
    'yolov7-d6': 166,
    'yolov7-e6': 144,
    'yolov7-e6e': 265,
}

def find_node(model, name):
    for idx, node in enumerate(list(model.graph.nodes)):
        if node.name.replace('module_dict_','') == name:
            return node
    return None

# Fusing BatchNorm with Convolutional layer
def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def _parent_name(target : str) -> Tuple[str, str]:
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op != 'call_module': 
            continue
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model

# Fusing BatchNorm with Convolutional layers
def rep_conv1x1(model, head,conv1x1,conv3x3,tail):
    head.users = {conv3x3:None}
    conv3x3.users = {tail:None}
    tail._input_nodes = {conv3x3:None}
    tail._args = (conv3x3,)

    modules = dict(model.named_modules())
    weight_1x1_expanded = torch.nn.functional.pad(modules[conv1x1.target].weight, [1, 1, 1, 1])
    rbr_1x1_bias = modules[conv1x1.target].bias
    modules[conv3x3.target].weight = torch.nn.Parameter(modules[conv3x3.target].weight + weight_1x1_expanded)
    modules[conv3x3.target].bias = torch.nn.Parameter(modules[conv3x3.target].bias + rbr_1x1_bias)

# Fusing Implicit Knowledge layers
def rep_im(model, head, target, tail_1, tail_2, im_a_weight, im_m_weight):
    # detach
    head.users = {target:None}
    target._input_nodes = {head:None}
    target._args = (head,)
    target.users = {tail_1:None,tail_2:None}

    tail_1._input_nodes = target
    tail_1._args = (target,tail_1._args[1])

    tail_2._input_nodes.pop(tail_2._args[0])
    tail_2._input_nodes[target] = None
    temp_arg = list(tail_2._args)
    temp_arg[0] = target
    temp_arg = tuple(temp_arg)
    tail_2._args = temp_arg 

    # reparam
    modules = dict(model.named_modules())

    conv_weight = modules[target.target].weight
    c2,c1,_,_ = conv_weight.shape
    c2_,c1_,_,_ = im_a_weight.shape

    # ImplicitA
    with torch.no_grad():
        modules[target.target].bias += torch.matmul(conv_weight.reshape(c2,c1), im_a_weight.reshape(c1_,c2_)).squeeze(1)

    # ImplicitM
    with torch.no_grad():
        modules[target.target].bias *= im_m_weight.reshape(c2)
        modules[target.target].weight *= im_m_weight.transpose(0,1)

def get_module_dict(model):
    while hasattr(model, 'module_dict'):
        model = model.module_dict
    return model

def fusing_yolov7(model,model_name, head_outs):
    imn = implicit_layer_number[model_name]
    model = model.float().eval()
    # fusing conv-bn
    print("Fusing Conv-BN")
    fused_model = fuse(model)
    print("Fusing Conv-BN - Success")

    print("Reparameterizing")
    # head_0
    if model_name == 'yolov7':
        head = find_node(fused_model,'model_75_act')
        conv1x1 = find_node(fused_model,'model_102_rbr_1x1_0')
        conv3x3 = find_node(fused_model,'model_102_rbr_dense_0')
        tail = find_node(fused_model,'model_102_act')
        rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)

        head = find_node(fused_model,'model_88_act')
        conv1x1 = find_node(fused_model,'model_103_rbr_1x1_0')
        conv3x3 = find_node(fused_model,'model_103_rbr_dense_0')
        tail = find_node(fused_model,'model_103_act')
        rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)
        
        head = find_node(fused_model,'model_101_act')
        conv1x1 = find_node(fused_model,'model_104_rbr_1x1_0')
        conv3x3 = find_node(fused_model,'model_104_rbr_dense_0')
        tail = find_node(fused_model,'model_104_act')
        rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)

    head = find_node(fused_model,f'model_{head_outs[0]}_act')
    target = find_node(fused_model,f'model_{imn}_m_0')
    tail_1 =  find_node(fused_model,'getattr_1')
    tail_2 = find_node(fused_model,'view')
    
    module_dict = get_module_dict(fused_model)
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['0']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['0']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)

    # head_1
    
    head = find_node(fused_model,f'model_{head_outs[1]}_act')
    target = find_node(fused_model,f'model_{imn}_m_1')
    tail_1 =  find_node(fused_model,'getattr_2')
    tail_2 = find_node(fused_model,'view_1')
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['1']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['1']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)


    # head_2

    head = find_node(fused_model,f'model_{head_outs[2]}_act')
    target = find_node(fused_model,f'model_{imn}_m_2')
    tail_1 =  find_node(fused_model,'getattr_3')
    tail_2 = find_node(fused_model,'view_2')
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['2']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['2']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)

    print("Reparameterizing - Success")

    print("Recompile...")
    fused_model.graph.lint()
    fused_model.recompile()
    print("Success")
    return fused_model
    
def fusing_yolov7_aux(model,model_name, head_outs):
    imn = implicit_layer_number[model_name]
    model = model.float().eval()
    # fusing conv-bn
    print("Fusing Conv-BN")
    fused_model = fuse(model)
    print("Fusing Conv-BN - Success")

    print("Reparameterizing")
    # head_0
    if model_name == 'yolov7':
        head = find_node(fused_model,'model_75_act')
        conv1x1 = find_node(fused_model,'model_102_rbr_1x1_0')
        conv3x3 = find_node(fused_model,'model_102_rbr_dense_0')
        tail = find_node(fused_model,'model_102_act')
        rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)

        head = find_node(fused_model,'model_88_act')
        conv1x1 = find_node(fused_model,'model_103_rbr_1x1_0')
        conv3x3 = find_node(fused_model,'model_103_rbr_dense_0')
        tail = find_node(fused_model,'model_103_act')
        rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)
        
        head = find_node(fused_model,'model_101_act')
        conv1x1 = find_node(fused_model,'model_104_rbr_1x1_0')
        conv3x3 = find_node(fused_model,'model_104_rbr_dense_0')
        tail = find_node(fused_model,'model_104_act')
        rep_conv1x1(fused_model,head,conv1x1,conv3x3,tail)

    head = find_node(fused_model,f'model_{head_outs[0]}_act')
    target = find_node(fused_model,f'model_{imn}_m_0')
    tail_1 =  find_node(fused_model,'getattr_1')
    tail_2 = find_node(fused_model,'view')
    
    module_dict = get_module_dict(fused_model)
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['0']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['0']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)

    # head_1
    
    head = find_node(fused_model,f'model_{head_outs[1]}_act')
    target = find_node(fused_model,f'model_{imn}_m_1')
    tail_1 =  find_node(fused_model,'getattr_2')
    tail_2 = find_node(fused_model,'view_2')
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['1']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['1']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)


    # head_2

    head = find_node(fused_model,f'model_{head_outs[2]}_act')
    target = find_node(fused_model,f'model_{imn}_m_2')
    tail_1 =  find_node(fused_model,'getattr_3')
    tail_2 = find_node(fused_model,'view_4')
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['2']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['2']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)


    head = find_node(fused_model,f'model_{head_outs[3]}_act')
    target = find_node(fused_model,f'model_{imn}_m_3')
    tail_1 =  find_node(fused_model,'getattr_4')
    tail_2 = find_node(fused_model,'view_6')
    im_a_weight = module_dict.model._modules[str(imn)]._modules['ia']._modules['3']._parameters['NOTA_implicit']
    im_m_weight = module_dict.model._modules[str(imn)]._modules['im']._modules['3']._parameters['NOTA_implicit']
    rep_im(fused_model,head,target,tail_1,tail_2,im_a_weight,im_m_weight)
    print("Reparameterizing - Success")

    print("Recompile...")
    fused_model.graph.lint()
    fused_model.recompile()
    print("Success")
    return fused_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/root/workspace/yolov7_training_models/yolov7w6_exported_l2norm_10.pt', help='finetuned-yolov7 graphmodule path')
    parser.add_argument('--model-name', type=str,default='yolov7-w6', choices=['yolov7','yolov7x','yolov7-d6','yolov7-e6','yolov7-e6e','yolov7-tiny','yolov7-w6'])
    parser.add_argument('--save-path', type=str, default='/root/workspace/yolov7_training_models/yolov7w6_exported_l2norm_10_rep_test_0418.pt', help='finetuned-yolov7 graphmodule path')
    parser.add_argument('--cfg', type=str, default='/root/workspace/NPMC_yolov7/cfg/training/yolov7-w6.yaml', help='model.yaml path')
    
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    head_outs = cfg['head'][-1][0]
    print(head_outs)
    ckpt = torch.load(args.model, map_location='cpu')

    if type(ckpt) == dict:
        model = ckpt['model']
    else:
        model = ckpt
    model.float().train()
    if args.model_name in ['yolov7','yolov7-tiny','yolov7x']:
        fused_model = fusing_yolov7(model, args.model_name, head_outs)
    elif args.model_name in ['yolov7-w6','yolov7-d6','yolov7-e6','yolov7-e6e']:
        fused_model = fusing_yolov7_aux(model, args.model_name, head_outs)
    else:
        NotImplementedError

    torch.save(fused_model,args.save_path)
    
    x = torch.randn(1,3,640,640)
    for a,b in zip(model(x),fused_model(x)):
        print(torch.allclose(a,b))
        print(torch.sum((a-b)**2))

# python reparameterization_npmc.py --model /root/workspace/NPMC_yolov7/compressed_traced_yolov7_training.pt --model-name yolov7 --save-path /root/workspace/NPMC_yolov7/compressed_traced_yolov7_training_rep_0418_test.pt --cfg /root/workspace/NPMC_yolov7/cfg/training/yolov7.yaml
# python reparameterization_npmc.py --model /root/workspace/yolov7_training_models/yolov7w6_exported_l2norm_10.pt --model-name yolov7-w6 --save-path /root/workspace/yolov7_training_models/yolov7w6_exported_l2norm_10_rep_test_0418.pt --cfg /root/workspace/NPMC_yolov7/cfg/training/yolov7-w6.yaml