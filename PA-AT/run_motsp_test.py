#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from train_motsp import get_angles

from options_motsp_TEST import get_options
from train_motsp import train_epoch, validate, get_inner_model
from reinforce_baselines_motsp import  RolloutBaseline, WarmupBaseline
from nets.attention_model_motsp_eval import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from nets.attention_model_motsp import set_decode_type

import scipy.io
import csv
import numpy as np
from utils import move_to
from time import time




def calc_hyp(pf,num_pf,dim,max_cts):
  
   num_pts = 100000
   pts = max_cts*np.random.rand(num_pts,dim)

   fdom = np.zeros((num_pts,),dtype=np.bool)
   lb = np.amin(pf,axis=0)
   #print(lb)
   #lba = np.ones((num_pts,dim))
   lba = np.tile(lb,(num_pts,1))


   a1 = pts > lba
   fcheck = np.all(a1,axis=1)

   for i in range(num_pf):
       if np.any(fcheck):
           #print(i)
           ar1 = pts[fcheck,:]
           l = ar1.shape[0]
           ar2 = np.tile(pf[i,:],(l,1))
           f1 = ar1 > ar2
           f = np.all(f1,axis=1)
           fdom[fcheck] = f
           fcheck[fcheck] = ~f


   hyp1 = np.array([100*(np.sum(fdom)/num_pts)])
   hyp = np.ndarray.astype(hyp1,np.float32)

   
   return hyp

def run(opts):

    torch.manual_seed(opts.seed)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    #file = open("pts.csv")
    #my_data = np.loadtxt(file, delimiter=",")





    # Figure out what's the problem
    problem = load_problem(opts.problem)
    opts.load_path = "outputs_motsp/motsp_40/motsp40_rollout_20220119T094543/epoch-4.pt"
    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)


    if True:

        pref_size = 2
        num_pref = 100
        model = AttentionModel(opts.embedding_dim, opts.hidden_dim, problem,pref_size,
                 num_pref, n_encode_layers=opts.n_encode_layers, mask_inner=True, mask_logits=True, normalization=opts.normalization,tanh_clipping=opts.tanh_clipping,checkpoint_encoder=opts.checkpoint_encoder,shrink_size=opts.shrink_size).to(opts.device)
        model_ = get_inner_model(model)
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
      
        pref, angs = get_angles(0,num_pref)
    
        prefa = torch.from_numpy(pref).float()
        prefa = prefa.to(opts.device)
        ri = opts.graph_size

        z2 = np.zeros((50,2))
        set_decode_type(model, "greedy")
        model.eval()
        filename = "kool_v1_hyp_abl_"+str(ri)+".mat"
        
        for j in range(1):
            z1 = np.zeros((100,4))
            
            file = open("test_data_iclr_csv/"+str(ri)+"/motsp"+str(ri)+"_"+str(j)+".csv")
            my_data = np.loadtxt(file, delimiter=",")
            a1 = np.reshape(my_data,[1,ri,5])
            a2 = a1[:,:,:-1]
            a3 = torch.from_numpy(a2.astype(np.float32))
            st = time()
            o1, o2, ob3, cstr, ll = model(prefa,move_to(a3, opts.device),np.ones((100,1,1)))
            et = time() - st
            #print(o1)
            #print(o2)

            for i in range(100):

                z1[i,0] = o1[i].cpu().numpy()[0]
                z1[i,1] = o2[i].cpu().numpy()[0]
                
                z1[i,2] = cstr[i].cpu().numpy()[0]

                #print('*********')
                #print("ob1"+ " " + str(z1[i,0]))
                #print("ob2"+ " " + str(z1[i,1]))
                #print("cstr"+ " " + str(z1[i,2]))
                #print('*********')
            if ri == 40:
                hyp = calc_hyp(z1[:,0:2],100,2,50)
            else:
                hyp = calc_hyp(z1[:,0:2],100,2,ri)

            z2[j,0] = hyp
            z2[j,1] = et
            print(hyp)
            print(et)
            #if j == 0:
            #    scipy.io.savemat("kool_motsp_"+str(ri)+"_0.mat", mdict={'C': z1})

        #scipy.io.savemat(filename, mdict={'C2': z2})



if __name__ == "__main__":
    run(get_options())
