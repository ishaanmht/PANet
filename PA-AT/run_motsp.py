#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from train_motsp import get_angles

from options_motsp import get_options
from train_motsp import train_epoch, validate, get_inner_model
from reinforce_baselines_motsp import  RolloutBaseline, WarmupBaseline
from nets.attention_model_motsp import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem

import scipy.io
import csv
import numpy as np


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)


    if opts.eval_only:

        pref_size = 2
        num_pref = 100
        model = AttentionModel(opts.embedding_dim, opts.hidden_dim, problem,pref_size,
                 num_pref, n_encode_layers=opts.n_encode_layers, mask_inner=True, mask_logits=True, normalization=opts.normalization,tanh_clipping=opts.tanh_clipping,checkpoint_encoder=opts.checkpoint_encoder,shrink_size=opts.shrink_size).to(opts.device)
        model_ = get_inner_model(model)
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

        pref, angs = get_angles(0,100)
        prefa = torch.from_numpy(pref).float()
        prefa = prefa.to(opts.device)

        z1 = np.zeros((100,4))

        ri = opts.graph_size
        file = open("test_data_2obj_iclr_csv/"+str(ri)+"/motsp"+str(ri)+"_"+str(0)+".csv")
        my_data = np.loadtxt(file, delimiter=",")
        a1 = np.reshape(my_data,[1,ri,5])
        a2 = a1[:,:,:-1]
        a3 = torch.from_numpy(a2.astype(np.float32))

        filename = "att_v1_ep7_"+str(ri)+".mat"
        o1, o2, cstr = validate(model, prefa, num_pref, a3, opts)
        print(o1)
        print(o2)

        for i in range(100):

            z1[i,0] = o1[i].cpu().numpy()[0]
            z1[i,1] = o2[i].cpu().numpy()[0]
            z1[i,2] = cstr[i].cpu().numpy()[0]

            print('*********')
            print("ob1"+ " " + str(z1[i,0]))
            print("ob2"+ " " + str(z1[i,1]))
            print("cstr"+ " " + str(z1[i,2]))
            print('*********')

        scipy.io.savemat(filename, mdict={'C': z1})
    

    
    print("Done")
    pref_size = 2
    num_pref = 20
    model = AttentionModel(opts.embedding_dim, opts.hidden_dim, problem,pref_size,
                 num_pref, n_encode_layers=opts.n_encode_layers, mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    baseline = RolloutBaseline(model, problem, opts)
   

    #if opts.bl_warmup_epochs > 0:
    #    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        pref, angs = get_angles(epoch_resume,num_pref)
        prefa = torch.from_numpy(pref).float()
        prefa = prefa.to(opts.device)
        baseline.epoch_callback(model, prefa, num_pref, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:

        pref, angs = get_angles(0,100)
        prefa = torch.from_numpy(pref).float()
        prefa = prefa.to(opts.device)

        z1 = np.zeros((N,4))

        ri = opts.graph_size
        file = open("test_data_2obj_iclr_csv/"+str(ri)+"/motsp"+str(ri)+"_"+str(j)+".csv")
        my_data = np.loadtxt(file, delimiter=",")
        a1 = np.reshape(my_data,[1,ri,5])
        a2 = a1[:,:,:-1]

        filename = "att_v1_ep7_"+str(ri)+".mat"
        o1, o2, cstr = validate(model, prefa, num_pref, a2.to(device), opts)
        print(o1)
        print(o2)

        for i in range(100):

            z1[i,0] = o1[i].cpu().numpy()[0]
            z1[i,1] = o2[i].cpu().numpy()[0]
            z1[i,2] = cstr[i].cpu().numpy()[0]

        scipy.io.savemat(filename, mdict={'C': z1})



    
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            print("*************************************************")
            train_epoch(
                model, pref_size,
                num_pref,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )



if __name__ == "__main__":
    run(get_options())
