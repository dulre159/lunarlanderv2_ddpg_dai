This repo contains my implementation of DQN and DDPG RL alghorithms.

The purpose of this code is limited to the project for my exam of DAI (Distributed Artificial Intelligence).

The scope was to test some exploration vs exploitation strategies for DDPG.

One can select from the main.py which strategy to use, set the number of training episodes and other params to test.

The environment that i used for testing was LunarLanderContinuous-v2 with total number of steps per episode limited to 300 to speed up the training.

MLP Neural Networks are used as function approximators. 

Networks structure is implements similarly to how it is in the original DDPG paper.

Layer Normalization is used instead of BatchNorm after each ReLU activation layer.
