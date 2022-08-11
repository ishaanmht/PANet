#-*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA


# Compute a sequence's reward
def reward(tsp_sequence):
    tour = np.concatenate((tsp_sequence, np.expand_dims(tsp_sequence[0],0))) # sequence to tour (end=start)
    inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1,:2]-tour[1:,:2]),axis=1)) # tour length
    return np.sum(inter_city_distances) # reward

# Swap city[i] with city[j] in sequence
def swap2opt(tsp_sequence,i,j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j+1] = np.flip(tsp_sequence[i:j+1], axis=0) # flip or swap ?
    return new_tsp_sequence

# One step of 2opt = one double loop and return first improved sequence
def step2opt(tsp_sequence):
    seq_length = tsp_sequence.shape[0]
    distance = reward(tsp_sequence)
    for i in range(1,seq_length-1):
        for j in range(i+1,seq_length):
            new_tsp_sequence = swap2opt(tsp_sequence,i,j)
            new_distance = reward(new_tsp_sequence)
            if new_distance < distance:
                return new_tsp_sequence, new_distance
    return tsp_sequence, distance


class DataGenerator(object):

    def __init__(self):
        
        self.ang = np.linspace(0.1744, 1.395, 100)
        pass

    def gen_instance(self, max_length, dimension, seed=0): # Generate random TSP instance
        if seed!=0: np.random.seed(seed)
        sequence = np.random.rand(max_length, dimension) # (max_length) cities with (dimension) coordinates in [0,1]
        pca = PCA(n_components=dimension) # center & rotate coordinates
        sequence = pca.fit_transform(sequence) 
        return sequence

    def train_batch(self, batch_size, max_length, dimension): # Generate random batch for training procedure
        input_batch = []
        for _ in range(batch_size):
            input_ = self.gen_instance(max_length, dimension) # Generate random TSP instance
            input_batch.append(input_) # Store batch
        return input_batch
    
    def gen_instance1(self, sequence, max_length, dimension, num_cities, seed=0): # Generate random TSP instance
        if seed!=0: np.random.seed(seed)
        seq = np.zeros((max_length,dimension+1))
        
        prob = np.arange(num_cities,dtype=np.float32)
        np.random.shuffle(prob)
        idx = np.where(prob == 0.0)
        prob = np.reshape(prob,(num_cities,1))
        prob2 = np.zeros((num_cities,1))
        prob2[0,0] = idx[0]
        
        #pca = PCA(n_components=dimension) # center & rotate coordinates
        #sequence = pca.fit_transform(sequence)
        c = np.concatenate((sequence,prob2), axis=1)
        seq[0:num_cities,:] = c
        return seq
        
    def gen_instancer(self, sequence, max_length, dimension, num_cities, seed=0): # Generate random TSP instance
        if seed!=0: np.random.seed(seed)
        seq = np.zeros((max_length,dimension+1))
        
        prob = np.arange(num_cities,dtype=np.float32)
        np.random.shuffle(prob)
        idx = np.where(prob == 0.0)
        prob[idx] = prob[24]
        prob[24] = 0.0
        prob = 0.001*np.reshape(prob,(num_cities,1))
        prob2 = np.zeros((num_cities,1))

        c = np.concatenate((sequence[0:num_cities,0:2],prob,prob2,prob2), axis=1)
        seq[0:num_cities,:] = c
        return seq
        

    def train_batch1(self, batch_size, max_length, dimension, num_cities): # Generate random batch for training procedure
        input_batch = []
        itr = range(batch_size)
        #ang2 = np.random.choice(self.ang)

        for i in itr:
            
            sequence = np.random.rand(num_cities, dimension+2) # (max_length) cities with (dimension) coordinates in [0,1]
                
            input_ = self.gen_instance1(sequence,max_length, dimension+2,num_cities) # Generate random TSP instance
            input_batch.append(input_) # Store batch
        
        return input_batch
    
    def train_batch_ro(self, batch_size, max_length, dimension, num_cities): # Generate random batch for training procedure
        input_batch = []
        itr = range(batch_size)
        #ang2 = np.random.choice(self.ang)

        for i in itr:
            
            sequence = np.random.rand(num_cities, dimension+1) # (max_length) cities with (dimension) coordinates in [0,1]
                
            input_ = self.gen_instancer(sequence,max_length, dimension+2,num_cities) # Generate random TSP instance
            input_batch.append(input_) # Store batch
        
        return input_batch

    def train_batch2(self, batch_size, max_length, dimension, num_cities): # Generate random batch for training procedure
        
        input_batch = []
        itr = range(batch_size)
            #ang2 = np.random.choice(self.ang)

        for i in itr:
            
            sequence = np.random.rand(num_cities, dimension+4) # (max_length) cities with (dimension) coordinates in [0,1]
                
            input_ = self.gen_instance1(sequence,max_length, dimension+4,num_cities) # Generate random TSP instance
            input_batch.append(input_) # Store batch
        
        return input_batch
    
    def test_batch1(self, batch_size, max_length, dimension, seed=0, shuffle=False): # Generate random batch for testing 
        input_batch = []
        input_ = self.gen_instance1(max_length, dimension, seed=seed) # Generate random TSP instance
        sequence = np.copy(input_)
        input_batch.append(sequence)
        input1 = input_[:,0:2]
        
        prob = np.ones((max_length, max_length))
        np.fill_diagonal(prob,0)
        c = np.concatenate((input1,prob), axis=1)
        sequence1 = np.copy(c)
        input_batch.append(sequence1)
        
        return input_batch

    
    def test_batch(self, batch_size, max_length, dimension, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        input_ = self.gen_instance(max_length, dimension, seed=seed) # Generate random TSP instance
        for _ in range(batch_size): 
            sequence = np.copy(input_)
            if shuffle==True: 
                np.random.shuffle(sequence) # Shuffle sequence
            input_batch.append(sequence) # Store batch
        return input_batch

    def loop2opt(self, tsp_sequence, max_iter=2000): # Iterate step2opt max_iter times (2-opt local search)
        best_reward = reward(tsp_sequence)
        new_tsp_sequence = np.copy(tsp_sequence)
        for _ in range(max_iter): 
            new_tsp_sequence, new_reward = step2opt(new_tsp_sequence)
            if new_reward < best_reward:
                best_reward = new_reward
            else:
                break
        return new_tsp_sequence, best_reward

    def visualize_2D_trip(self, trip): # Plot tour
        plt.figure(1)
        colors = ['red'] # First city red
        for i in range(len(trip)-1):
            colors.append('blue')
            
        plt.scatter(trip[:,0], trip[:,1],  color=colors) # Plot cities
        tour=np.array(list(range(len(trip))) + [0]) # Plot tour
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--")

        plt.xlim(-0.75,0.75)
        plt.ylim(-0.75,0.75)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def visualize_sampling(self, permutations): # Heatmap of permutations (x=cities; y=steps)
        max_length = len(permutations[0])
        grid = np.zeros([max_length,max_length]) # initialize heatmap grid to 0

        transposed_permutations = np.transpose(permutations)
        for t, cities_t in enumerate(transposed_permutations): # step t, cities chosen at step t
            city_indices, counts = np.unique(cities_t,return_counts=True,axis=0)
            for u,v in zip(city_indices, counts):
                grid[t][u]+=v # update grid with counts from the batch of permutations

        fig = plt.figure(1) # plot heatmap
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(grid, interpolation='nearest', cmap='gray')
        plt.colorbar()
        plt.title('Sampled permutations')
        plt.ylabel('Time t')
        plt.xlabel('City i')
        plt.show()
