# -*- coding: utf-8 -*-

import torch

def L21_norm(L21_norm_input):
    '''
    Function L21_norm(L21_norm_input)
    To apply L21_norm function on the input 2D-tensor.
    
    Parameters:
        L21_norm_input: 2D-tensor
    
    Return:
        L21_norm_result: torch.float
    '''
    L21_norm_result = torch.norm(L21_norm_input, dim = 1).sum()
    return L21_norm_result