#!/usr/bin/env python

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



dataset = DataGenerator() # Create Data Generator


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)



import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

parser1 = argparse.ArgumentParser()
parser1.add_argument("--num_cities", "-ct", help="number of cities")
args1 = parser1.parse_args()
ri = int(args1.num_cities)

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

def str2bool(v):
  return v.lower() in ('true', '1')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=1, help='batch size')
data_arg.add_argument('--pref_size', type=int, default=1, help='batch size')
data_arg.add_argument('--max_length', type=int, default=ri, help='number of cities') ##### #####
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
train_arg.add_argument('--is_training', type=str2bool, default=False, help='switch to inference mode when model is trained')
data_arg.add_argument('--start_city', type=int, default=False, help='Do you provide start City?')
data_arg.add_argument('--scaling_factor', type=int, default=1000, help='Scaling Factor for constraint')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed



config, _ = get_config()
config.is_training = False
config.start_city = True
config.batch_size = 1
config.max_length = ri
config.temperature = 1.2

tf.reset_default_graph()
actor = Actor(config) # Build graph


variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)   

#Generate TSP Instance
np.random.seed(1)
input_batch = dataset.train_batch1(1, actor.max_length,2,actor.max_length)
a1 =input_batch[0]
a1[0,4] = 24.0 #
input_batch[0] = a1

#Select Preferences
ang = np.reshape(np.linspace(0.174, 1.48, 100),(1,100))
r_x =  np.cos(ang)
r_y =  np.sin(ang)
r_a = np.append(r_x,r_y,0)
r_a = np.reshape(r_a.T,(100,1,2))


#Filenames For output storage
filename = "Out_objs_"+str(config.max_length)+".csv"
model_dir = "PA_Net_model"

with tf.Session() as sess:  # start session

    sess.run(tf.global_variables_initializer()) # Run initialize op
    pr = 100
    rt = 1
    saver.restore(sess, model_dir+"/save_0/actor.ckpt")
    z1 = np.zeros((pr,rt))
    z2 = np.zeros((pr,rt))
    z3 = np.zeros((pr,rt))
    z4 = np.zeros((pr,rt))
    for i in range(pr):#10
        
        r_b = np.reshape(r_a[i,:,:],(1,1,2))
        feed = {actor.input1: input_batch,   actor.ang_pref: r_b}
        
        for k in range(rt):
        
            t1 = time.time()
            tour,cstr1, dis1, dis2, dis3 = sess.run([actor.tour ,  actor.constr,  actor.dist1, actor.dist2, actor.dist3], feed_dict=feed)
            t2 = time.time() - t1
            
     
            print("++++++++++")
            print(dis1)
            print(dis2)
            print("++++++++++")
            
            
           
        
            z1[i,k] = dis1
            z2[i,k] = dis2
            z3[i,k] = cstr1
           
            
            
        row_c = [i, np.mean(z1[i,:]),np.mean(z2[i,:])]
        append_list_as_row(filename, row_c)
    
    
  
    plt.plot(np.mean(z1,axis=1),np.mean(z2,axis=1),'o')
    plt.show()
            
        
        

    







