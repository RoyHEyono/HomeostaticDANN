from danns_eg.data.cifar import get_cifar_dataloaders
from danns_eg.data.mnist import get_sparse_mnist_dataloaders
import torch
import json
from munch import DefaultMunch
import danns_eg.resnets as resnets
from torch.cuda.amp import GradScaler, autocast
import tqdm
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import numpy as np
from torch.optim import lr_scheduler, Adam
from danns_eg.optimisation import AdamW, get_linear_schedule_with_warmup, SGD
import danns_eg.optimisation as optimizer_utils
import torch.nn.functional as F
import danns_eg.utils as train_utils
import copy
import argparse


# Load model
# Compute gradient for all weights: G
# Zero out all gradients after the layer
# Update the weights before the layer
# Compute gradient for all weights: G'


def get_optimizer(p, model):
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
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
                   momentum=p.opt.momentum) #,exponentiated_grad=p.opt.exponentiated) 
        opt.nesterov = p.opt.nesterov
        # opt.eg_normalise = p.opt.eg_normalise
        return opt

    elif p.opt.algorithm.lower() == "adamw":
        #  this should be adapted in future for adamw specific args! 
        return AdamW(parameters, lr = p.opt.lr,
                     weight_decay=p.opt.wd,
                     exponentiated_grad=p.opt.exponentiated) 

def eval_model(epoch, model, loaders, loss_fn_sum, p):
    model.eval()
    with torch.no_grad():
        train_correct, n_train, train_loss, train_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['train_eval']:
            with autocast():
                num_of_local_layers = 0
                out = model(ims)
                loss_val = loss_fn_sum(out, labs)
                train_loss += loss_val
                #print(f"Global Loss: {loss_val.item()}")
                train_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_train += ims.shape[0]
                loc_loss = 0.0
                for name, param in model.named_parameters():
                    if 'local_loss_value' in name:
                        if param.item() > 0:
                            loc_loss += param.item()
                            num_of_local_layers += 1
                if num_of_local_layers > 0:
                    loc_loss /= num_of_local_layers
                    train_local_loss += loc_loss
        train_acc = train_correct / n_train * 100
        train_loss /=  n_train
        train_local_loss /= n_train
        
        test_correct, n_test, test_loss, test_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['test']:
            with autocast():
                # out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                num_of_local_layers = 0
                out = model(ims)
                loss_val = loss_fn_sum(out, labs)
                test_loss += loss_val
                #print(f"Global Loss: {loss_val.item()}")
                test_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_test += ims.shape[0]
                loc_loss = 0.0
                for name, param in model.named_parameters():
                    if 'local_loss_value' in name:
                        if param.item() > 0:
                            loc_loss += param.item()
                            num_of_local_layers += 1

                if num_of_local_layers > 0:
                    loc_loss /= num_of_local_layers
                    test_local_loss += loc_loss

        #print("Not currently running train eval!")
        test_acc = test_correct / n_test * 100
        test_loss /=  n_test
        test_local_loss /= n_test

        results["test_accs"].append(test_acc)
        results["test_losses"].append(test_loss.cpu().item())
        results["train_accs"].append(train_acc)
        results["train_losses"].append(train_loss.cpu().item())
        results["train_local_losses"].append(train_local_loss)
        results["test_local_losses"].append(test_local_loss)
        results["ep_i"].append(epoch)

def grad_update(model, loss, scaler, homeostasis: bool):
    
    if homeostasis:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'Wix' in name or 'Wei' in name:
                    continue
                
                param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0]
                
                # if 'Wex' in name:
                #     param.grad = None # This locks in the homeostatic loss only
    else:
        scaler.scale(loss).backward()


def train_epoch(model, loaders, loss_fn, opt, scheduler, p, scaler, ics_layer):
    # the vars below should all be defined in the global scope
    epoch_correct, total_ims = 0, 0
    layer_grad = ics_layer
    for batch_idx, (ims, labs) in enumerate(loaders['train']):
        G = {}
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(ims)
            loss = loss_fn(out, labs)
            
            # TODO: We need to return the local loss from the model
            # local_loss = model(ims).ln_mu_loss

            batch_correct = out.argmax(1).eq(labs).sum().cpu().item()
            batch_acc = batch_correct / ims.shape[0] * 100
            epoch_correct += batch_correct
            total_ims += ims.shape[0]

        weights = model.state_dict()

        grad_update(model, loss, scaler, p.model.homeostasis)

        # Store the gradients - G
        for name, param in model.named_parameters():
            layer = int(name.split('.')[1])
            if param.requires_grad:
                G[name] = param.grad
                # Clear the gradient for all layers past (i)_th layer
                if layer >= layer_grad:
                    param.grad = None


        
        scaler.step(opt)
        scaler.update()

        

        # ##################################################
        # # Start of ICS calculation
        # ##################################################

        with autocast():
            out_secondpass = model(ims)
            loss_secondpass = loss_fn(out_secondpass, labs)

        if p.model.homeostasis:
            grad_update(model, loss_secondpass, scaler, p.model.homeostasis)
        else:
            loss_secondpass.backward()

        for name, param in model.named_parameters():
            layer = int(name.split('.')[1])
            if param.requires_grad:
                if layer == layer_grad and batch_idx == 0:
                    bsz = G[name].shape[0]
                    # Check for NaN values
                    has_nan = torch.isnan(G[name]).any() or torch.isnan(param.grad).any()

                    # Check for Inf values
                    has_inf = torch.isinf(G[name]).any() or torch.isinf(param.grad).any()

                    compute_cosine = not (has_nan or has_inf)

                    if name in G_prime:
                        G_prime[name]['mse'].append(F.mse_loss(G[name], param.grad).item())
                        if compute_cosine:
                            G_prime[name]['cosine'].append(F.cosine_similarity(G[name].reshape(bsz,-1), param.grad.reshape(bsz,-1), dim=1).mean(dim=0).item())
                    else:
                        G_prime[name] = {'mse':list(), 'cosine':list()}
                        G_prime[name]['mse'].append(F.mse_loss(G[name], param.grad).item())
                        if compute_cosine:
                            G_prime[name]['cosine'].append(F.cosine_similarity(G[name].reshape(bsz,-1), param.grad.reshape(bsz,-1), dim=1).mean(dim=0).item())

        
        ##################################################
        # Reload model weights
        ##################################################
        model.load_state_dict(weights)

        # ##################################################
        # # Reset to original gradients
        # ##################################################
        for name, param in model.named_parameters():
            layer = int(name.split('.')[1])
            if param.requires_grad:
                param.grad = G[name]

        scaler.step(opt)
        scaler.update()
        try: scheduler.step()
        except NameError: pass

        # ##################################################
        # # End of ICS calculation
        # ##################################################
                    

    epoch_acc = epoch_correct / total_ims * 100
    results["online_epoch_acc"] = epoch_acc

def get_dataloaders(p):
    if "cifar" in p.train.dataset : return get_cifar_dataloaders(p) 
    elif "mnist" in p.train.dataset : return get_sparse_mnist_dataloaders(p)
    else:print(f"ERROR: {p.train.dataset} not recognised as a vaild dataset")

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A simple script with a flag")

    # Add a flag to the ArgumentParser
    parser.add_argument("-s", "--seed", type=int, help="Seed", default=2)
    parser.add_argument("-c", "--cfg_file", type=str, help="config_filename", default="model_resnet9_lr_0.1_wd_1e-06_m_0.9_norm_None_epochs_50_seed_5_lagrangian_False")
    parser.add_argument("-l", "--ics_layer", type=int, help="Layer for ICS metric", default=3)


    # Parse the command-line arguments
    args = parser.parse_args()

    print(f"We're at: {os.getcwd()}")

    # Load json file
    # model_resnet9_lr_0.1_wd_1e-06_m_0.9_norm_ln_epochs_50_seed_5_lagrangian_False
    # model_resnet9_lr_0.1_wd_1e-06_m_0.9_norm_None_epochs_50_seed_5_lagrangian_False
    # model_resnet9_lr_0.1_wd_0.001_m_0.9_norm_ln_epochs_50_seed_5_lagrangian_True
    cfg_filename = args.cfg_file
    with open(f'result_data/{cfg_filename}', 'r') as f:
        model_json = json.load(f)

    # Load model weights
    cfg = DefaultMunch.fromDict(model_json)
    new_seed = args.seed
    cfg.train.seed = new_seed
    model = resnets.resnet9_kakaobrain(cfg).cuda()
    train_utils.set_seed_all(cfg.train.seed)
    # model.load_state_dict(torch.load('model_resnet50_lr_0.1_wd_0.0001_m_0.9_norm_ln_epochs_50_seed_0_lagrangian_True.pt'))

    results = {"test_accs" :[], "test_losses": [], 
                "train_accs" :[], "train_losses":[], "ep_i":[],
                "model_norm":[], "train_local_losses":[], "test_local_losses":[]}
    G_prime = {}

    loaders = get_dataloaders(cfg)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_sum = CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
    opt = get_optimizer(cfg, model)
    scaler = GradScaler()
    EPOCHS = cfg.train.epochs
    iters_per_epoch = len(loaders['train'])
    lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                            [0, 1, 0])
    
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

    progress_bar = tqdm(range(1,1+cfg.train.epochs))

    log_epochs = 1
        
    for epoch in progress_bar:
        train_epoch(model, loaders, loss_fn, opt, scheduler, cfg, scaler, args.ics_layer)  
        if epoch%log_epochs==0:
            eval_model(epoch, model, loaders, loss_fn_sum, cfg)    
            progress_bar.set_description("Train/test accuracy after {} epochs: {:.2f}/{:.2f}".format(epoch,results["train_accs"][-1],results["test_accs"][-1]))

    # Print best test and training accuracy
    print("Best train accuracy: {:.2f}".format(max(results["train_accs"])))
    print("Best test accuracy: {:.2f}".format(max(results["test_accs"])))

    G_prime["test_accs"] = results["test_accs"]

    # Save dictionary G_prime here
    g_prime_filename = f'ics_data/{cfg_filename}_G_Prime_new_seed_{new_seed}_ics_layer_{args.ics_layer}.json'
    g_prime_json = json.dumps(G_prime)
    with open(g_prime_filename, "w") as f:
        f.write(g_prime_json)

    