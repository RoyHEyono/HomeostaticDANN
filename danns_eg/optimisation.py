"""
Links to reference implementations
http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
https://github.dev/huggingface/transformers/tree/main/src/transformers/models


Expected that the main function will create the optimiser with a function similar to the
following example:
```
def get_optimizer(param_groups, ):
    params_groups = get_param_groups(p, model)
    # first construct the iterable of param groups
    parameters = []
    for k, group in params_groups.items():
        d = {"params":group, "name":k}
        
        if k in ["wix_params", "wei_params",'wex_params']: 
            d["positive_only"] = True
        else: 
            d["positive_only"] = False
        
        if p.opt.use_sep_inhib_lrs:
            if k == "wix_params": d['lr'] = p.opt.inhib_lrs.wix
            elif k == "wei_params": d['lr'] = p.opt.inhib_lrs.wei
        
        if k == "norm_biases":
            d['exponentiated_grad'] = False 
        if p.opt.use_sep_bias_gain_lrs:
            if k == "norm_biases": 
                d['lr'] = p.opt.bias_gain_lrs.b
                print("hard coding non exp grad for biases")
            elif k == "norm_gains": d['lr'] = p.opt.bias_gain_lrs.g
        
        parameters.append(d)
    
    if p.opt.algorithm.lower() == "sgd":
        opt = SGD(parameters, lr = p.opt.lr,
                   weight_decay=p.opt.wd,
                   momentum=p.opt.momentum,
                   exponentiated_grad=p.opt.exponentiated) 
        opt.nesterov = p.opt.nesterov
        opt.eg_normalise = p.opt.eg_normalise
        return opt

    elif p.opt.algorithm.lower() == "adamw":
        #  this should be adapted in future for adamw specific args! 
        return AdamW(parameters, lr = p.opt.lr,
                     weight_decay=p.opt.wd,
                     exponentiated_grad=p.opt.exponentiated) 

```

"""
import math
from typing import Callable, Iterable, Optional, Tuple, Union
from numpy import double

import torch
from torch import nn, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_param_groups(model, return_groups_dict=False):
    """
    Utility function to seperate model parameters into groups.

    Unless return_groups_dict=True, returns a list of dictionaries in
    the format expected by pytorch optimisers, ie:
    [ {'params': [],'name':str },  {'params':[], 'name':str, 'lr': float}],

    If return_groups_dict is True, returns a dictionary of name:param_list pairs. 
    """
    norm_layers = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d) # update this if needed
    param_groups = {
        "norm_biases":[],"norm_gains":[], "bias_params":[],
        "wix_params":[], "wei_params": [],'wex_params':[],
        "rho_params":[], "other_params":[] }
    
    for name, m in model.named_modules():
        if len(list(m.named_parameters(recurse=False))) == 0:
            continue # skip modules that do not have child parameters 
        
        if isinstance(m,  norm_layers):
            param_groups['norm_biases'].append(m.bias)
            param_groups['norm_gains'].append(m.weight)
            continue
        
        for k, param in m.named_parameters(recurse=False):
            if k.lower().endswith("ix"): param_groups['wix_params'].append(param)
            elif k.lower().endswith("ei"): param_groups['wei_params'].append(param)
            elif k.lower().endswith("ex"): param_groups['wex_params'].append(param)
            elif "bias" in k.lower(): param_groups['bias_params'].append(param)
            elif "rho"  in k.lower():param_groups['rho_params']
            else: 
                param_groups['other_params'].append(param)

    # drop empty lists (for e.g if not a dann no ex, ix, ei etc)
    param_groups = {k:l for k,l in param_groups.items() if len(l)> 0}

    # check we have every parameter
    all_params = [] 
    for group in param_groups.values(): 
        all_params+=group
    all_params = set(all_params)
    for k, param in model.named_parameters():
        assert param in all_params

    if return_groups_dict:
        return param_groups
    # construct the list as expected by pytorch optimisers
    param_groups_list = []
    for k, group in param_groups.items():
        param_groups_list.append({"params":group, "name":k})
    return param_groups_list
    
class SGD(Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=0.0, momentum=0.9,
                 update_algorithm: str = "gd", 
                 weight_decay_algorithm: str = "same", 
                 positive_only: bool = False,
                 normalise_weights: bool = False,
                 nesterov: bool = False ):
        """
        Args:
            params : Iterable of parameters or list of dictionaries defining parameter groups.
            lr :
            weight_decay:
            momentum: note this is the torch sgd version of momentum, i.e lr is applied to update v.
                      Also note that in my impl. weight decay is decoupled from the momentum. 
                      (This seems to be slightly better than pytorch's impl but not carefully checked) 
            update_algorithm: str in {"gd", "eg"}
            weight_decay_algorithm : str in {"same", "gd", "eg"}. If same (default), uses the same alg as 
                                    "update_alg".
            positive_only : bool, if true clamps params min 0. 
            normalise_weights : whether to normalise the sum of weights after an update
            nesterov: whether to use nesterov momentum. 

        Todo:
            A nice extension would be clamp the signs on the weights.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        update_algorithm=update_algorithm, 
                        weight_decay_algorithm=weight_decay_algorithm,
                        positive_only=positive_only,
                        normalise_weights=normalise_weights, nesterov=nesterov)
        super().__init__(params, defaults)
        
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                mu = group["momentum"]
                if mu > 0:
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["v"] = torch.zeros_like(p.data) # initialise momentum buffer
                    
                    state["v"].mul_(mu).add_(p.grad.data)
                    state["step"] += 1 # not sure if we want to track this
                    
                    if group["nesterov"]:
                        p.grad += mu * state["v"]
                    else: 
                        p.grad = state["v"]
                else:
                    pass

                if group["normalise_weights"]:
                    state = self.state[p] 
                    try: c = state["c"]
                    except KeyError: 
                        state["c"] = torch.sum(p.data*p.data.sign()).cpu().numpy()
                        c = state["c"]
                
                # Update params using grad attr
                if group["update_algorithm"] == "eg":
                    p.data.mul_(torch.exp(p.sign() * p.grad.data * -group["lr"]))        
                elif group["update_algorithm"] == "gd": # normal gradient descent update
                    p.data.add_(p.grad.data, alpha=-group["lr"])
                else: 
                    print("Error update algorithm should be one of ['eg', 'gd']")
                    raise
                
                # Weight decay update
                if group["weight_decay"] > 0.0:
                    wd_step_size = -group["lr"] * group["weight_decay"]
                    
                    if group["weight_decay_algorithm"] == "same":
                        group["weight_decay_algorithm"] = group["update_algorithm"]
                    assert group["weight_decay_algorithm"] in ["gd", "eg"] 
                    
                    if group["weight_decay_algorithm"] == "eg":
                        p.data.mul_(torch.exp(p.data.sign() * p.data * wd_step_size))
                    elif group["weight_decay_algorithm"] == "gd":   
                        p.data.add_(p.data, alpha=wd_step_size)
                
                # Project params to postives
                try: 
                    if group["positive_only"]: p.data.clamp_(min=0)
                except: pass 

                if  group["normalise_weights"]: 
                    div_factor = c/torch.sum(p.data*p.data.sign()).cpu().numpy()
                    p.data.mul_(div_factor)
            


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    https://github.dev/huggingface/transformers/tree/main/src/transformers/models

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        decoupled_weight_decay (`bool`, defaults to `True`):
            Implement decoupled weight decay as in the AdamW paper, or standard
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        exponentiated_grad: bool = False,
        wd_decoupled:bool = True, # choose to decouple weight decay - to implement
        # check maths but should be able to just change the step size for wd update
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, 
                        correct_bias=correct_bias,
                        exponentiated_grad=exponentiated_grad,
                        wd_decoupled=wd_decoupled)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if group["exponentiated_grad"]:
                    p.data.mul_(torch.exp(p.sign() * p.grad.data * -group["lr"]))
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    if group["exponentiated_grad"]:
                        wd_step_size = -group["lr"] * group["weight_decay"]
                        p.data.mul_(torch.exp(p.sign() * p.data * wd_step_size))
                    else:
                        p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

                # clip parameters positive if required (eg danns) 
                try: 
                    if group["positive_only"]: p.data.clamp_(min=0)
                except: 
                    # we want to throw the error
                    if group["positive_only"]: p.data.clamp_(min=0)
                    pass 


        return loss

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)