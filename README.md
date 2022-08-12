# Pareto Frontier Approximation Network (PA-Net) to Solve Bi-objective TSP

This repositry contains the code for our work on solving bi-Objective TSP (BTSP) through Reinforcement learning. We present Pareto frontier approximation network (PA-Net), a network that generates good approximations of the Pareto front for the bi-objective travelling salesperson problem (BTSP). In our approach, BTSP is converted into a constrained optimization problem. We then train our network to solve this constrained problem using the Lagrangian relaxation and policy gradient. With PA-Net we improve the performance over an existing deep RL-based method. The average improvement in hypervolume metric which is used to measure optimality of the Pareto front is 2.3%.

We use existing TSP networks for our by objective extension. The code for each network can be found in the folders PA-AT and PA-EAN.

Our project website: https://sites.google.com/view/pa-net-btsp

Our Paper: https://arxiv.org/pdf/2203.01298.pdf

Our work is accepted at IEEE CASE 2022. If you find our work helpful, please cite our paper.
