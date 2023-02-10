import math
from pprint import pprint
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binom
from scipy.optimize import minimize

def generate_noisy_r_of_k_data(n_datapoints, n, k, r, class_balance=0.5, 
                               p_rel=0.5, p_irrel="same", seed=None, shuffle=False, 
                               noise_probs=None, w_star=None, neg_wstar=False, 
                               verbose=False):
    """
    Slightly more complicated version of generate_r_of_k_data func
    in the "learning_r_of_k_functions" module.

    Note y is now <w^*,x> >=r, so replace with r=1 for 1 of k. Before
    we used strictly greater than r 

        p_rel : the binomial draw prob of the relevant datapoints
                if none, we will seek to estimate it from class balance, k, r.
        if p_rel is None:
        p_rel = estimate_prel(k,r,class_balance)
        noise_probs: vector to use to draw from {-1,0,1} for adding noise

        neg_wstar_elements: bool, if true w_star will be drawn with negative
                            elements.
    
    Note this function is used for two types of data generation. The noisy version
    of learning r of k (with -1,0, 1 noise.) And the neg u vector, mean 0. So in this
    case threshold is 0 and we can draw 0.5 on at any time directly? (no need to estimate)

    """
    if seed is not None: np.random.seed(seed)
    assert k <= n
    if k % 2: print("WARNING: Intended to run with k being even for data balance ")
    if p_irrel.lower() == "same": p_irrel=p_rel
        
    # Generate data
    if verbose: print(f"Drawing data with probability {p_rel}")
    X_k = np.random.binomial(n=1, p=p_rel,   size=(n_datapoints,k))
    X_n = np.random.binomial(n=1, p=p_irrel, size=(n_datapoints,n-k))
    X = np.concatenate([X_k, X_n], axis=1)
    X.astype(float)
    # Generate "target" vector, w_star
    if w_star is None:
        w_star = np.concatenate([np.ones(k), np.zeros(n-k)], axis=0)
        if neg_wstar:
            neg_inds = np.random.choice(range(k), size=int(k/2), replace =False)
            print(int(k/2), neg_inds)
            w_star[neg_inds] =  w_star[neg_inds]*-1
            print("w_star: ", w_star[:20])

    if shuffle: # This is cosmetic only
        shuffle_idxs = np.random.permutation(n)
        w_star = w_star[shuffle_idxs]
        if verbose:
            X = X[:,shuffle_idxs]

    if noise_probs is None:
        if verbose: print(" drawing data without noise")
        y = np.matmul(X,w_star) >= r
    else:
        y_true = np.matmul(X,w_star) >= r
        noise_vec = np.random.choice([-1,0,1],size=X.shape[0],p=noise_probs)
        y = (np.matmul(X,w_star) + noise_vec) >=r
        w_star_acc = sum(y_true == y) / X.shape[0]
        # in future maybe work set noise based on chosen error rate
        print(np.mean(np.matmul(X,w_star)))
        if verbose:
            print(f" adding noise {-1,0,1} with prob. {noise_probs}")
            print("w_star_acc is", w_star_acc)
            print(f"% of positive y_true_s drawn: {y_true.sum()/len(y_true)}")
            print(f"% of positive y drawn: {y.sum()/len(y)}")

    y = y.astype(int)
    return X, y, w_star

# --------------------------------------------------------------------------------------
# Legacy functions for generating r of k data with positive only u vector below.
# --------------------------------------------------------------------------------------
def get_binom_sf_vs_balance_min_func(n_trials, threshold, balance):
    """
    Returns a func to be minimised as a function of p (prob of trial 
    success) such that P( # successes > threshold) ~= balance.
    
    n_trials corresponds to k, the # of relevant data dimensions.
    """
    def f(p):
        """
        Returns the squared distance of the binomial dist 
        survival function (which is 1-cdf) from the desired class balance. 
        
        Args : p, the probability of success
        """
        return (binom(n_trials, p).sf(x=threshold) - balance)**2

    return f 

def generate_r_of_k_data(n_datapoints, n, k, r, class_balance=0.5, 
                         p_irrel="same", seed=None, shuffle=False):
    """
    Generates data of the form y = 1 if r < sum_{i \in K} x_i
    where K is a set indexing k < n "relevant" dimensions of x.  

    See section 2.3.2 of multiplicative update notes:
         https://www.overleaf.com/2484566799kchhkjgxkhdn

    Args:
        n_datapoints
        k : # of relevant dimensions (int)
        n : # of input dimensions
        r : threshold, # of k relevant vars required for for y=1
        class_balance : balance of positive classes, used to estimate p_rel
                        which is the probability task relevant data dims will 
                        be 0 or 1.
        p_irrel : "same" or  \in [0,1]. This is the probability
                  task irrelevant data dimensions will be 1 or 0. 
                  If "same" we use the same p as for the k relevant.
    """
    if seed is not None: np.random.seed(seed)
    assert k <= n

    # Estimate a p_rel for desired class balance
    f = get_binom_sf_vs_balance_min_func(n_trials=k, threshold=r, balance=class_balance)
    opt_res = minimize(f, 0.1, bounds=[[0,1]])
    
    if opt_res["success"] == False:
        print("Something went wrong estimating p_rel!")
        print(opt_res)
        raise
    else:
        p_rel = opt_res["x"][0]
        if p_irrel.lower() == "same": p_irrel=p_rel
        
    # Generate data
    #print(f"Drawing data with probability {p_rel}")
    X_k = np.random.binomial(n=1, p=p_rel,   size=(n_datapoints,k))
    X_n = np.random.binomial(n=1, p=p_irrel, size=(n_datapoints,n-k))
    X = np.concatenate([X_k, X_n], axis=1)
    X.astype(float)
    # Generate "target" vector, w_star
    w_star = np.concatenate([np.ones(k), np.zeros(n-k)], axis=0)

    if shuffle: # I don't think this adds anything...
        shuffle_idxs = np.random.permutation(n)
        w_star = w_star[shuffle_idxs]
        X = X[:,shuffle_idxs]

    y = np.matmul(X,w_star) > r # strictly great 
    y = y.astype(int)
    return X, y

def estimate_prel(k,r,class_balance):
    # Estimate a p_rel for desired class balance
    # but why do use a threshold of r-1?! not r? 
    # be aware in the winnow code that r should probs be r as > not >=
    # ut just change to >= 
    # 
    f = get_binom_sf_vs_balance_min_func(n_trials=k, threshold=r-1, balance=class_balance)
    opt_res = minimize(f, 0.1, bounds=[[0,1]])
    if opt_res["success"] == False:
        print("Something went wrong estimating p_rel!")
        print(opt_res)
        raise
    else:
        p_rel = opt_res["x"][0]
    
    return p_rel

