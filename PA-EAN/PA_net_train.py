#!/usr/bin/env python

'''
Code to Train PA-Net for 2 objective MOTSP. PA-Net uses the architecture of network presented in :
Learning Heuristics for the TSP by Policy Gradient, Deudon M., Cournut P., Lacoste A., Adulyasak Y. and Rousseau L.M.
The code of PA-Net is modified version of the code presented in
github link : https://github.com/MichelDeudon/encode-attend-navigate
'''
import tensorflow as tf
import time
distr = tf.contrib.distributions
import numpy as np
import os
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from Actor import Actor
import csv
import argparse

parser = argparse.ArgumentParser(description='Configuration file')
parser1 = argparse.ArgumentParser()
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


dataset = DataGenerator() # Create Data Generator
epochs = 1
tt = 20

data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=60, help='batch size')
data_arg.add_argument('--pref_size', type=int, default=tt, help='batch size')
data_arg.add_argument('--max_length', type=int, default=100, help='number of cities') ##### #####
data_arg.add_argument('--dimension', type=int, default=2, help='city dimension')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_embed', type=int, default=128, help='actor critic input embedding')
net_arg.add_argument('--num_neurons', type=int, default=512, help='encoder inner layer neurons')
net_arg.add_argument('--num_stacks', type=int, default=3, help='encoder num stacks')
net_arg.add_argument('--num_heads', type=int, default=16, help='encoder num heads')
net_arg.add_argument('--query_dim', type=int, default=360, help='decoder query space dimension')
net_arg.add_argument('--num_units', type=int, default=256, help='decoder and critic attention product space')
net_arg.add_argument('--num_units_pref', type=int, default=512, help='decoder and critic attention product space')
net_arg.add_argument('--num_neurons_critic', type=int, default=256, help='critic n-1 layer')

# Train / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_steps', type=int, default=10, help='nb steps')
train_arg.add_argument('--init_B', type=float, default=0.01, help='critic init baseline')
train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr_decay_step', type=int, default=20000, help='lr1 decay step')
train_arg.add_argument('--lr_decay_rate', type=float, default=0.50, help='lr1 decay rate')
train_arg.add_argument('--temperature', type=float, default=1.0, help='pointer initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer tanh clipping')
train_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode when model is trained')
data_arg.add_argument('--start_city', type=int, default=False, help='Do you provide start City?')
data_arg.add_argument('--scaling_factor', type=int, default=1000, help='Scaling Factor for constraint')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def get_angles(n):
    ang1 = np.linspace(0.174, 1.48, 100)
    ang  = np.random.choice(ang1,(1,n))
    r_x =  np.cos(ang)
    r_y =  np.sin(ang)
    r_a = np.append(r_x,r_y,0)
    r_a = np.reshape(r_a.T,(n,1,2))
    return r_a
    

ang = np.reshape(np.linspace(0.174, 1.48, tt),(1,tt))

r_x =  np.cos(ang)
r_y =  np.sin(ang)
r_a = np.append(r_x,r_y,0)
r_a = np.reshape(r_a.T,(tt,1,2))

config, _ = get_config()
dir1 = "PA_Net_model_1"

print(dir)
lamn = 0.001
lamd = 0.0000125
lamm = 25.0


angs = np.reshape(ang,(config.pref_size,1,1))
lam1 = lamn*np.ones((config.pref_size,1,1))


tf.reset_default_graph()
actor = Actor(config) # Build graph





variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)   




with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    variables_names = [v.name for v in tf.trainable_variables() if 'Adam' not in v.name]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
       
        pass




id1 = 0
id2 = 0
np.random.seed(123)
tf.set_random_seed(123)


with tf.Session() as sess: # start session
    save_path = dir + '/save'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    sess.run(tf.global_variables_initializer()) # run initialize op
    writer = tf.summary.FileWriter(dir_+'/summary', sess.graph) # summary writer

    
    for j in range(epochs):
    
    
        
        if j > 0:
            lamm = 1.5*lamm
            lamd = 1.5*lamd
            
        for i in range(int(config.nb_steps)): # Forward pass & train step
        
            if i > 19999 and i%1000 == 0:
              
                lamd = 0.00125
                r_a = get_angles(tt)
               
                lam1 = lamn*np.ones((config.pref_size,1,1))
            
            input_batch = dataset.train_batch1(actor.batch_size, actor.max_length,2,actor.max_length)
            feed = {actor.input1: input_batch, actor.i: ri, actor.itr: i, actor.ang_pref: r_a, actor.lam1: lam1} # get feed dict
            loss1_a, loss1_c, pred1_c, lgr, cstr, dis1, dis2, dis3, summary, _, _ = sess.run([actor.loss, actor.loss21,  actor.predictions, actor.reward, actor.constr, actor.dist1, actor.dist2, actor.dist3, actor.merged, actor.trn_op1, actor.trn_op2], feed_dict=feed)
            
            
            for k in range(config.pref_size):
            
                cr1 = np.mean(cstr[k*config.batch_size:(k+1)*config.batch_size])
                
           

                lam1[k,0,0] = lam1[k,0,0] + lamd*cr1
            

                if lam1[k,0,0] > lamm:
                    lam1[k,0,0] = lamm
                

                if lam1[k,0,0] < 0:
                    lam1[k,0,0] = 0.0001


        saver.save(sess, save_path+"_"+str(j)+"/actor.ckpt")

