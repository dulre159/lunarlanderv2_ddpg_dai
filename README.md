<p align="center">
<img src="https://github.com/dulre159/LunarLanderV2-DDPG-DAI/blob/master/lunarlanderv2gif.gif" alt="LunarLanderv2 environment">
</p>

<h1>Strategies for solving OpenAIGym's LunarLanderV2 with Deep Deterministic Policy Gradient</h1>
This repo contains my implementation of <link> Deep Q Learning (DQN) and Deep Deterministic Policy Gradient (DDPG) Reinforcement Learning (RL) algorithms for solving the <a href="https://www.gymlibrary.dev/environments/box2d/lunar_lander/">LunarLanderV2</a> problem.</br>
This code was written for a university exam, and the scope was to test some exploration vs exploitation strategies for DDPG agents.

<h2>You can read more about RL, DQN, and DDPG here</h2>
<a href="https://spinningup.openai.com/en/latest/spinningup/rl_intro.html">Part 1: Key Concepts in RL</a></br>
<a href="https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html">Part 2: Kinds of RL Algorithms</a></br>
<a href="https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html">Part 3: Intro to Policy Optimization</a></br>
<a href="https://www.tensorflow.org/agents/tutorials/0_intro_rl?hl=en">Introduction to RL and Deep Q Networks</a></br>
<a href="https://spinningup.openai.com/en/latest/algorithms/ddpg.html">Deep Deterministic Policy Gradient</a></br>

<h2>Requirements</h2>
<a href="https://pypi.org/project/numpy/">numpy</a></br>
<a href="https://pytorch.org/">pytorch</a></br>
<a href="https://github.com/openai/gym">gym</a></br>

<h2>USAGE</h2>
The code is set to work with the OpenAIGym's LunarLanderV2 environment.</br>
In the main.py file you can set the strategy to use, the number of training episodes and tune other params.</br>
The total number of steps per episode is limited to 300 to speed up the training.</br>

<h2>Details</h2>
A Multilayer Perceptron (MLP) is used as a function approximator</br>
The actor and critic networks are implemented as described in the original DDPG paper. (citations are below)</br>
Layer Normalization is used instead of BatchNorm after each ReLU activation layer.</br>

</br>
</br>

```
@misc{lillicrap2019continuouscontroldeepreinforcement,
      title={Continuous control with deep reinforcement learning},
      author={Timothy P. Lillicrap and Jonathan J. Hunt and Alexander Pritzel and Nicolas Heess and Tom Erez and Yuval Tassa and David Silver and Daan Wierstra},
      year={2019},
      eprint={1509.02971},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1509.02971},
}

@misc{mnih2013playingatarideepreinforcement,
      title={Playing Atari with Deep Reinforcement Learning},
      author={Volodymyr Mnih and Koray Kavukcuoglu and David Silver and Alex Graves and Ioannis Antonoglou and Daan Wierstra and Martin Riedmiller},
      year={2013},
      eprint={1312.5602},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1312.5602}, 
}

@misc{ba2016layernormalization,
      title={Layer Normalization},
      author={Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton},
      year={2016},
      eprint={1607.06450},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1607.06450},
}

```
