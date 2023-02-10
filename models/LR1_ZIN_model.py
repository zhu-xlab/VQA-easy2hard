#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhenghang
"""

from torchvision import models as torchmodels
import torch.nn as nn
import models.seq2vec
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import skimage.io as io
import pdb

VISUAL_OUT = 2048
QUESTION_OUT = 2400
FUSION_IN = 1200

FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

import math

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)    
    output = torch.matmul(scores, v)
    return scores, output

class SumAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(SumAttention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, mid_features, 1)
        #self.x_conv2 = nn.Conv2d(mid_features, mid_features, 1)
        self.att_conv = nn.Conv2d(mid_features, 1, 1)
        self.mid_features = mid_features
        self.norm = Norm((mid_features,8,8))
        #self.norm2 = Norm((mid_features,8,8))

        self.linear_vout = nn.Linear(256*4,1200)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        vo = self.v_conv(v)
        q = self.q_lin(q)
        q = tile_2d_over_nd(q, vo)
        v = v / (torch.norm(vo, dim=1, p=2, keepdim=True) + 1e-5)
        q = q / (torch.norm(q, dim=1, p=2, keepdim=True) + 1e-5)
        cmf = v + q
        xc = self.relu(cmf)
        x = self.x_conv(xc)
        cmfx = x
        x = x.view(x.shape[0],256,-1).permute([0,2,1])
        v = v.view(v.shape[0],256,-1).permute([0,2,1])
        amap, v_out = attention(x, v, x, self.mid_features)
        amap = amap.sum(dim=1)
        v_out = torch.nn.functional.avg_pool2d(v_out, kernel_size=4)
        v_out = self.linear_vout(v_out.view(-1, 256*4))
        return amap, v_out, cmfx
        

def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


# Region Learning

class VQAModel(nn.Module):
    def __init__(self, vocab_questions, vocab_answers, input_size = 512):
        super(VQAModel, self).__init__()
        
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        self.seq2vec = models.seq2vec.factory(self.vocab_questions, {'arch': 'skipthoughts', 'dir_st': 'data/skip-thoughts', 'type': 'BayesianUniSkip', 'dropout': 0.25, 'fixed_emb': False})
        for param in self.seq2vec.parameters():
            param.requires_grad = False
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        
        self.visual = torchmodels.resnet152(pretrained=True)
        extracted_layers = list(self.visual.children())
        extracted_layers = extracted_layers[0:8] #Remove the last fc and avg pool
        self.visual = torch.nn.Sequential(*(list(extracted_layers)))
        for param in self.visual.parameters():
            param.requires_grad = False
        
        output_size = (input_size / 32)**2
        self.visual_spatial = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(256),1))
        self.visual = torch.nn.Sequential(self.visual, torch.nn.Conv2d(2048,int(2048/output_size),1))
        self.linear_v = nn.Linear(VISUAL_OUT, FUSION_IN)
        
        self.linear_av = nn.Linear(1024*2, FUSION_IN)
                
        self.global_attention = SumAttention(256, 1200, 256, 1)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
    
    
    def forward(self, input_v, input_q):       
        x_q = self.seq2vec(input_q)
        x_q = self.dropoutV(x_q)
        x_q = self.linear_q(x_q)
        x_sq = x_q
        x_q = nn.Tanh()(x_q)   #1200

        x_sv = self.visual_spatial(input_v) # 70, 256, 8, 8
                
        global_amap, v_out, sim_vq = self.global_attention(x_sv, x_sq)
                
        
        x_ov = self.visual(input_v)

                
        x_ov = x_ov.view(-1, VISUAL_OUT)
        
        x_v = self.dropoutV(x_ov)
        x_v = self.linear_v(x_v)
        x_v = nn.Tanh()(x_v)
        
        x_v = v_out + x_v
        
        x = torch.mul(x_v, x_q)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x, global_amap
        


if __name__=='__main__':
    rr = RegionRegressor()
    cmf = torch.randn([70,256,8,8])
    cmf_out = rr(cmf)
    print(cmf_out.size())
