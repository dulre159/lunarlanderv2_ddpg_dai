This repo contains my implementation of DQN and DDPG RL alghorithms for DAI (Distributed Artificial Intelligence) project for the exam.
The scope was to test some exploration vs exploitation strategies for DDPG.
One can select from the main which strategy to use, set the number of training episodes and other params to test.
The environment that i used for testing was LunarLanderContinuous-v2 with totale numper of steps per episode limited to 300 to speed up the training.
MLP Neural Networks are used as function approximators. 
Networks structure is implements similarly to how it is in the original DDPG paper.
Layer Normalization is used instead of BatchNorm.
