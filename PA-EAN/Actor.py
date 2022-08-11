# -*- coding: utf-8 -*-
'''
 PA-Net uses the architecture of network presented in :
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
from utils_mup import embed_seq, encode_seq, full_glimpse, pointer, embed_pref


class Actor(object):
    
    
    def __init__(self,config):
        
        self.batch_size = config.batch_size # batch size
        self.batch_pref = config.pref_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.dimension = config.dimension + 2  # dimension of a city (coordinates)
        self.start_city = config.start_city
        self.scaling_factor = config.scaling_factor
        self.C = config.C
        self.temperature = config.temperature
        # Network config
        self.input_embed = config.input_embed # dimension of embedding space
        self.num_neurons = config.num_neurons # dimension of hidden states (encoder)
        self.num_stacks = config.num_stacks # encoder num stacks
        self.num_heads = config.num_heads # encoder num heads
        self.query_dim = config.query_dim # decoder query space dimension
        self.num_units = config.num_units # dimension of attention product space (decoder and critic)
        self.num_units_pref = config.num_units_pref # dimension of attention product space (decoder and critic)
        self.num_neurons_critic = config.num_neurons_critic # critic n-1 layer num neurons
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
       
        
        # Training config (actor and critic)
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # actor global step
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2") # critic global step
        self.init_B = config.init_B # critic initial baseline
        self.lr_start = config.lr_start # initial learning rate
        self.lr_decay_step = config.lr_decay_step # learning rate decay step
        self.lr_decay_rate = config.lr_decay_rate # learning rate decay rate
        self.is_training = config.is_training # swith to False if test mode
        
        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input1 = tf.placeholder(tf.float32, [None, self.max_length, self.dimension+1], name="input_coordinates")
        self.ang_pref = tf.placeholder(tf.float32, [None, 1 , 2], name="input_coordinates")
        self.lam1 = tf.placeholder(tf.float32, [None, 1 , 1], name="input_coordinates")
        
        
        self.input_ = self.input1[:,:,:-1]
        self.i = tf.placeholder(tf.int64, name="r")
        self.itr = tf.placeholder(tf.int64, name="itr")
        
        
        
        with tf.variable_scope("actor"): self.encode_decode()
        with tf.variable_scope("critic"): self.build_critic()
        with tf.variable_scope("environment"): self.build_reward()
        with tf.variable_scope("optimizer"): self.build_optim()
        self.merged = tf.summary.merge_all()
        
        
    def encode_decode(self):
        actor_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        
        actor_pref = embed_pref(pref=self.ang_pref, from_=2, to_= self.input_embed, is_training=self.is_training, num_units=self.input_embed, BN=True, initializer=self.initializer) # batch_pref X 1 X Num_units_pref
        #actor_pref = tf.tile(tf.reshape(actor_pref,[1,1,self.num_units_pref]),tf.constant([self.batch_size,self.max_length,1], tf.int32))

        actor_encoding_1 = encode_seq(input_seq=actor_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
                
        
        if self.is_training == False:
            actor_encoding_1 = tf.tile(actor_encoding_1,[self.batch_size,1,1])
        
       
        n_hidden = actor_encoding_1.get_shape().as_list()[2] # input_embed
        
        
        W_ref = tf.get_variable("W_ref",[1, n_hidden, self.num_units],initializer=self.initializer)
        W_q = tf.get_variable("W_q",[self.query_dim, self.num_units],initializer=self.initializer)
        v = tf.get_variable("v",[self.num_units],initializer=self.initializer)
        
        
       
     
        W_1 =tf.get_variable("W_1",[n_hidden, self.query_dim],initializer=self.initializer) # update trajectory (state)
        W_2 =tf.get_variable("W_2",[n_hidden, self.query_dim],initializer=self.initializer)
        W_3 =tf.get_variable("W_3",[n_hidden, self.query_dim],initializer=self.initializer)
        
        
        
        
        for i in range(self.batch_pref):
            
            
            actor_encoding = actor_encoding_1 + tf.tile(tf.reshape(actor_pref[i],[1,1,self.input_embed]),tf.constant([self.batch_size,self.max_length,1], tf.int32))
            encoded_ref = tf.nn.conv1d(actor_encoding, W_ref, 1, "VALID") # actor_encoding is the ref for actions [Batch size, seq_length, n_hidden]
            
            idx_list, log_probs, entropies = [], [], [] # tours index, log_probs, entropies
            mask = tf.zeros((self.batch_size, self.max_length))
            
            query1 = tf.zeros((self.batch_size, n_hidden)) # initial state
            query2 = tf.zeros((self.batch_size, n_hidden)) # previous state
            query3 = tf.zeros((self.batch_size, n_hidden)) # previous previous state
            c = self.max_length
            if self.start_city == True:
            
                query = tf.nn.relu(tf.matmul(query1, W_1) + tf.matmul(query2, W_2) + tf.matmul(query3, W_3))
                logits = pointer(encoded_ref=encoded_ref, query=query, mask=mask, W_ref=W_ref, W_q=W_q, v=v, C=self.C, temperature=self.temperature)
                prob = distr.Categorical(logits) # logits = masked_scores
                a = self.input1[:,:,-1]
                b = tf.reduce_sum(a,axis=1)
                idx = tf.dtypes.cast(b, tf.int32)
                
                idx_list.append(idx) # tour index
                log_probs.append(prob.log_prob(idx)) # log prob
                entropies.append(prob.entropy()) # entropies
                mask = mask + tf.one_hot(idx, self.max_length) # mask
                
                idx_ = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch
                query3 = query2
                query2 = query1
                query1 = tf.gather_nd(actor_encoding, idx_) # update trajectory (state)
                c = c-1
            
            
            
            
            
            for step in range(c): # sample from POINTER
                
                query = tf.nn.relu(tf.matmul(query1, W_1) + tf.matmul(query2, W_2) + tf.matmul(query3, W_3))
                logits = pointer(encoded_ref=encoded_ref,  query=query, mask=mask, W_ref=W_ref, W_q=W_q, v=v, C=self.C, temperature=self.temperature)
                prob = distr.Categorical(logits) # logits = masked_scores
                idx = prob.sample()
                
                idx_list.append(idx) # tour index
                log_probs.append(prob.log_prob(idx)) # log prob
                entropies.append(prob.entropy()) # entropies
                mask = mask + tf.one_hot(idx, self.max_length) # mask
                
                idx_ = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch
                query3 = query2
                query2 = query1
                query1 = tf.gather_nd(actor_encoding, idx_) # update trajectory (state)
            
            
            idx_list.append(idx_list[0]) # return to start
            
            if i == 0 :
                self.tour = tf.expand_dims(tf.stack(idx_list, axis=1), axis=0) # permutations
                print(self.tour)
                self.log_prob = tf.add_n(log_probs) # corresponding log-probability for backprop
                self.entropies = tf.add_n(entropies)
            else:
                a1 = tf.expand_dims(tf.stack(idx_list, axis=1), axis=0) # permutations
                self.tour = tf.concat([self.tour,a1], axis=0)
                self.log_prob = tf.concat([self.log_prob,tf.add_n(log_probs)], axis=0)
                self.entropies = tf.concat([self.entropies,tf.add_n(entropies)], axis=0)
        
        print(self.log_prob)
        
        tf.summary.scalar('log_prob_mean', tf.reduce_mean(self.log_prob))
        tf.summary.scalar('entropies_mean', tf.reduce_mean(self.entropies))
        
        
        
    
    def build_reward(self): # reorder input % tour and return tour length (euclidean distance)
        
        for i in range(self.batch_pref):
            
            tour = self.tour[i]
            permutations = tf.stack([tf.tile(tf.expand_dims(tf.range(self.batch_size,dtype=tf.int32),1),[1,self.max_length+1]),tour],2)
            ordered_input_ = tf.gather_nd(self.input_,permutations)
            if self.is_training==True:
                ordered_input_ = tf.gather_nd(self.input_,permutations)
            else:
                ordered_input_ = tf.gather_nd(tf.tile(self.input_,[self.batch_size,1,1]), permutations)
            
            ordered_input_ = tf.transpose(ordered_input_,[2,1,0]) # [features, seq length +1, batch_size]   Rq: +1 because end = start
            ordered_x1_ = ordered_input_[0] # ordered x, y coordinates [seq length +1, batch_size]
            ordered_y1_ = ordered_input_[1] # ordered y coordinates [seq length +1, batch_size]
            ordered_x2_ = ordered_input_[2]
            ordered_y2_ = ordered_input_[3]
            
            delta_x2_1 = tf.transpose(tf.square(ordered_x1_[1:]-ordered_x1_[:-1]),[1,0]) # [batch_size, seq length]        delta_x**2
            delta_y2_1 = tf.transpose(tf.square(ordered_y1_[1:]-ordered_y1_[:-1]),[1,0]) # [batch_size, seq length]        delta_y**2
            inter_city_distances_1 = tf.sqrt(delta_x2_1+delta_y2_1) # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city: depot --> ... ---> depot      [batch_size, seq length]
            
            delta_x2_2 = tf.transpose(tf.square(ordered_x2_[1:]-ordered_x2_[:-1]),[1,0]) # [batch_size, seq length]        delta_x**2
            delta_y2_2 = tf.transpose(tf.square(ordered_y2_[1:]-ordered_y2_[:-1]),[1,0]) # [batch_size, seq length]        delta_y**2
            inter_city_distances_2 = tf.sqrt(delta_x2_2+delta_y2_2) # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city: depot --> ... ---> depot      [batch_size, seq length]
            
            d0 = tf.ones([self.batch_size],tf.float32)
            #d1 = tf.cos(self.ang_pref[i,0,0]*d0)
            #d2 = tf.sin(self.ang_pref[i,0,0]*d0)
            
            dist1  = tf.cast(tf.reduce_sum(inter_city_distances_1, axis=1), tf.float32) # [batch_size]
            dist2= tf.cast(tf.reduce_sum(inter_city_distances_2, axis=1), tf.float32) # [batch_size]t
            dist3 = tf.sqrt(dist1 * dist1 + dist2*dist2)
            r3_3 = tf.divide(self.ang_pref[i,0,0]*dist1+self.ang_pref[i,0,1]*dist2,dist3)
            
            
            constr = self.scaling_factor*tf.subtract(1.0,r3_3)
            reward = tf.add(dist3, self.lam1[i,0,0]*constr)
            
            if i == 0:
                self.constr = constr
                self.reward = reward
                self.dist1 = dist1
                self.dist2 = dist2
                self.dist3 = dist3
            
            else:
            
                self.constr = tf.concat([self.constr, constr],axis=0)
                self.reward = tf.concat([self.reward,reward],axis=0)
                self.dist1 = tf.concat([self.dist1,dist1],axis=0)
                self.dist2 = tf.concat([self.dist2,dist2],axis=0)
                self.dist3 = tf.concat([self.dist3,dist3],axis=0)
        
        print(self.constr)

                       
            
                
                
        
        
        
        
    def build_critic(self):
        critic_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        critic_pref = embed_pref(pref=self.ang_pref, from_=2, to_= self.input_embed, is_training=self.is_training, num_units=self.input_embed, BN=True, initializer=self.initializer)
        #critic_pref = tf.tile(tf.reshape(critic_pref,[1,1,self.num_units]),tf.constant([self.batch_size,self.max_length,1], tf.int32))
        critic_encoding_1 = encode_seq(input_seq=critic_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
        
        from_=self.input_embed
        to_=self.num_units
        W_ref_c = tf.get_variable("W_ref_c",[1, self.input_embed , self.num_neurons_critic],initializer=self.initializer)
        w1 = tf.get_variable("w1", [self.num_neurons_critic, 1], initializer=self.initializer)
        b1 = tf.Variable(self.init_B, name="b1")
        
        W_ref_g =tf.get_variable("W_ref_g",[1,from_, to_],initializer=self.initializer)
        W_q_g =tf.get_variable("W_q_g",[from_, to_],initializer=self.initializer)
        v_g =tf.get_variable("v_g",[to_],initializer=self.initializer)
        
        
        for i in range(self.batch_pref):
            
            critic_encoding = critic_encoding_1 +  tf.tile(tf.reshape(critic_pref[i],[1,1,self.input_embed]),tf.constant([self.batch_size,self.max_length,1], tf.int32))
            frame = full_glimpse(ref=critic_encoding, W_ref_g = W_ref_g, W_q_g=W_q_g,v_g=v_g, from_=self.input_embed, to_=self.num_units, initializer=tf.contrib.layers.xavier_initializer()) # Glimpse on critic_encoding [Batch_size, input_embed]
            
            h0 = tf.nn.relu(tf.matmul(frame, W_ref_c))
            
            if i == 0:
                self.predictions = tf.matmul(h0, w1)+b1
            else:
            
                self.predictions = tf.concat([self.predictions, tf.matmul(h0, w1)+b1], axis =0)
        

      

        

            
            
    def build_optim(self):
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): # Update moving_mean and moving_variance for BN
        
            with tf.name_scope('reinforce'):
            
                lr1 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.itr, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate1") # learning rate actor
                tf.summary.scalar('lr', lr1)
                opt1 = tf.train.AdamOptimizer(learning_rate=lr1) # Optimizer
            
                for i in range(self.batch_pref):
                    if i == 0:
                        self.loss = tf.reduce_mean(tf.stop_gradient(tf.squeeze(self.reward[i*self.batch_size:(i+1)*self.batch_size])- tf.squeeze(self.predictions[i]))*tf.squeeze(self.log_prob[i*self.batch_size:(i+1)*self.batch_size]), axis=0) # loss1 actor
                    else:
                        self.loss = self.loss+ tf.reduce_mean(tf.stop_gradient(tf.squeeze(self.reward[i*self.batch_size:(i+1)*self.batch_size])- tf.squeeze(self.predictions[i]))*tf.squeeze(self.log_prob[i*self.batch_size:(i+1)*self.batch_size]), axis=0)
                
                
            
                gvs1 = opt1.compute_gradients(self.loss) # gradients
                cgvs1 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs1 if grad is not None] # L2 clip
               
            
                
                self.trn_op1 = opt1.apply_gradients(grads_and_vars=cgvs1) # minimize op actor
            
            with tf.name_scope('state_value'):
            
                lr2 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.itr, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate2") # learning rate critic
                opt2 = tf.train.AdamOptimizer(learning_rate=lr2) # Optimizer
                
                for i in range(self.batch_pref):
                    
                    p1 = tf.reshape(self.reward[i*self.batch_size:(i+1)*self.batch_size],[self.batch_size,1])
                    if i == 0:
                        
                        self.loss21 = tf.losses.mean_squared_error(p1, self.predictions[i]) # loss critic
                    
                    else:
                
                        self.loss21 = self.loss21 + tf.losses.mean_squared_error(p1, self.predictions[i]) # loss critic
                
                
                
                gvs21 = opt2.compute_gradients(self.loss21)# gradients
                cgvs21 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs21 if grad is not None] # L2 clip
                
                
                self.trn_op2 = opt2.apply_gradients(grads_and_vars=cgvs21) # minimize op critic
        








