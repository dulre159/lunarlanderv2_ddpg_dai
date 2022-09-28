This repo contains my implementation of DQN and DDPG RL algorithms.

The purpose of this code is limited to the project for my exam of DAI (Distributed Artificial Intelligence).

The scope was to test some exploration vs exploitation strategies for DDPG.

One can select from the file main.py which strategy to use, set the number of training episodes and tune other params.

The code is set to work with the OpenAIGym's LunarLanderV2 environment.

The total number of steps per episode are limited to 300 to speed up the training.

MLP Neural Networks are used as function approximators. 

The actor critic networks are implemented as described in the original DDPG's paper.

Layer Normalization is used instead of BatchNorm after each ReLU activation layer.
