import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model_motsp import set_decode_type
from utils.log_utils_motsp import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def get_angles(epoch, n):
    
    if epoch == 0:
        
        ang = np.reshape(np.linspace(0.174, 1.48, n),(1,n))
        r_x =  np.cos(ang)
        r_y =  np.sin(ang)
        r_a = np.append(r_x,r_y,0)
        r_a = np.reshape(r_a.T,(n,2,1))

    else:
    
        ang1 = np.linspace(0.174/2, 1.48, int(0.5*n + 1))
        for i in range(1,11):
            anga = np.linspace(ang1[i-1], ang1[i],5)
            if i == 1:
                ang  = np.random.choice(anga,2, replace=False)
            else:
                ang = np.append(ang,np.random.choice(anga,2, replace=False),0)

        ang = np.sort(ang)
        ang = np.reshape(ang,(1,n))
        r_x =  np.cos(ang)
        r_y =  np.sin(ang)
        r_a = np.append(r_x,r_y,0)
        r_a = np.reshape(r_a.T,(n,2,1))
    
    return r_a, ang



def validate(model, pref, num_pref, dataset, opts):
    # Validate
    print('Validating...')
    ob1, ob2, cst = rollout_eval(model, pref, num_pref, dataset, opts)

    return ob1, ob2, cst


def rollout(model,  pref, num_pref, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    
    def eval_model_bat(pref,bat, num_pref):
        with torch.no_grad():
            ob1, ob2, ob3, cstr, ll = model(pref,move_to(bat, opts.device),np.ones((num_pref,1,1)))
            
            for i in range(num_pref):
                
                if i == 0:
                    at = ob3[i]
                else:
                    at = at + ob3[i]
            at = (1/num_pref)*at
            


        return at.data.cpu()

    tt = torch.cat([
        eval_model_bat(pref,bat, num_pref)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

    return tt

def rollout_eval(model,  pref, num_pref, data, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    
    with torch.no_grad():
        ob1, ob2, ob3, cstr, ll = model(pref,move_to(data, opts.device),np.ones((num_pref,1,1)))
                   
    return ob1, ob2, cstr


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, pref_size, num_pref, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    pref, angs = get_angles(epoch,num_pref)
    prefa = torch.from_numpy(pref).float()
    prefa = prefa.to(opts.device)
    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution), prefa, num_pref)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    
    lamn = 0.001
    lamd = 0.0000125
    lam_max = 25.0
    lamm = lamn*np.ones((num_pref,1,1))

   

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        
        
        lamm_new = train_batch(
            model, num_pref, prefa,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            lamn, lamd, lam_max,
            lamm,
            tb_logger,
            opts
        )

        lamm = lamm_new
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, prefa, num_pref, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, prefa, num_pref, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model, num_pref, pref,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        lamn, lamd, lam_max,
        lamm,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    ob1, ob2, ob3, cstr, ll = model(pref,x,lamm)

    # Evaluate baseline, get baseline loss if any (only for critic)

    bl_ob1, bl_ob2, bl_ob3, bl_cstr = baseline.eval(pref,x,lamm, 0)
    bl_loss = 0
    # Calculate loss
    o1,o2,o3,cst,ll_l = [],[],[],[],[]
    
    for i in range(num_pref):
        log_likelihood = ll[i]
        
        if i == 0:
            reinforce_loss = ((ob3[i] - bl_ob3[i]) * log_likelihood).mean()
        else:
            reinforce_loss = reinforce_loss + ((ob3[i] - bl_ob3[i]) * log_likelihood).mean()
    
        ct = cstr[i].mean()
        lamm[i,0,0] = lamm[i,0,0] + lamd*ct
        
        if lamm[i,0,0] < lamn:
            lamm[i,0,0] = lamn
        if lamm[i,0,0] > lam_max:
            lamm[i,0,0] = lam_max
        
        o1.append(ob1[i].mean().item())
        o2.append(ob2[i].mean().item())
        o3.append(ob3[i].mean().item())
        cst.append(ct.item())
        ll_l.append(-log_likelihood.mean().item())


    # Perform backward pass and optimization step
    optimizer.zero_grad()
    reinforce_loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values([o1[0], o2[0], cst[0], o3[0], ll_l[0]], [o1[4], o2[4], cst[4], o3[4], ll_l[4]], [o1[8], o2[8], cst[8], o3[8],ll_l[8]], [o1[15], o2[15], cst[15], o3[15], ll_l[15]], grad_norms, epoch, batch_id, step,
                    reinforce_loss, bl_loss, tb_logger, opts)
    return lamm
