import math
from collections import deque
import random
import numpy as np
import torch
from torch import nn

from DDPGActorCriticNetworks import ActorNetwork, CriticNetwork
import os.path

class DDPGAgent():
    def __init__(self, env, actor_model_name, actor_target_model_name, critic_model_name, critic_target_model_name, replay_memory,
                 exp_exp_strategy_name="", load_models_from_disk=False, minibatch_size=256, discount_rate=0.99, tau=1e-2,
                 actor_learning_rate=1e-04, critic_learning_rate=1e-03, eps_init=1.0, eps_min=0.01, eps_dec=0.0009,
                 memory_size=100000, gnoisestd = 0.1, eps_vdbe_mt_lt=1, eps_vdbe_mt_lt_mt_tau=2, eps_vdbe_mt_lt_lt_tau=2,
                 eps_vdbe_mt_lt_boltz_tau=5):

        self.exp_exp_strategy_name = exp_exp_strategy_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.os_shape = env.observation_space.shape
        self.os_high = env.observation_space.high
        self.os_low = env.observation_space.low
        self.os_high_tensor = torch.from_numpy(self.os_high).float().unsqueeze(0).to(self.device)
        self.os_low_tensor = torch.from_numpy(self.os_low).float().unsqueeze(0).to(self.device)

        self.as_shape = env.action_space.shape
        self.as_high = env.action_space.high
        self.as_low = env.action_space.low
        self.as_high_tensor = torch.from_numpy(self.as_high).float().unsqueeze(0).to(self.device)
        self.as_low_tensor = torch.from_numpy(self.as_low).float().unsqueeze(0).to(self.device)

        self.actor_model_name = actor_model_name
        self.actor_target_model_name = actor_target_model_name
        self.critic_model_name = critic_model_name
        self.critic_target_model_name = critic_target_model_name
        self.eps_dec = eps_dec
        self.eps = eps_init
        self.discount_rate = discount_rate
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.eps_min = eps_min
        self.memory_size = memory_size
        self.replay_memory = deque(maxlen=self.memory_size) if replay_memory is None else replay_memory
        self.minibatch_size = minibatch_size
        self.load_models_from_disk = load_models_from_disk
        self.tau = tau
        self.gnoisestd = gnoisestd
        #self.gnoisestd = torch.abs(torch.max(self.as_high_tensor)).item()
        #self.gnoisestd = torch.abs(torch.max(self.as_high_tensor)).item()/2
        self.eps_vdbe_mt_lt = eps_vdbe_mt_lt
        self.eps_vdbe_mt_lt_mt_tau = eps_vdbe_mt_lt_mt_tau
        self.eps_vdbe_mt_lt_lt_tau = eps_vdbe_mt_lt_lt_tau
        self.eps_vdbe_mt_lt_boltz_tau = eps_vdbe_mt_lt_boltz_tau
        self.eps_vdbe_mt_lt_teta = 1 / self.as_shape[0]

        self.build_models()

    def build_models(self):
        self.actor_model = ActorNetwork(self.actor_learning_rate, self.os_shape[0], 200, 150, self.as_shape[0], self.actor_model_name)
        self.actor_target_model = ActorNetwork(self.actor_learning_rate, self.os_shape[0], 200, 150, self.as_shape[0], self.actor_target_model_name)

        self.critic_model = CriticNetwork(self.critic_learning_rate, self.os_shape[0], 200, 150, self.as_shape[0], self.critic_model_name)
        self.critic_target_model = CriticNetwork(self.critic_learning_rate, self.os_shape[0], 200, 150, self.as_shape[0], self.critic_target_model_name)

        if self.load_models_from_disk and os.path.isfile(self.critic_model_name) and os.path.isfile(self.critic_target_model_name) and os.path.isfile(self.actor_model_name) and os.path.isfile(self.actor_target_model_name):
            self.actor_model.load_checkpoint()
            self.actor_target_model.load_checkpoint()
            self.critic_model.load_checkpoint()
            self.critic_target_model.load_checkpoint()

        self.actor_model.to(self.device)
        self.actor_target_model.to(self.device)
        self.critic_model.to(self.device)
        self.critic_target_model.to(self.device)

        self.critic_loss = nn.MSELoss()

    def save_state(self, state):
        observation, action, reward, next_observation, done = state
        self.replay_memory.append([observation, action, reward, next_observation, done])

    # def get_mod_sigmoid_noise(self, avg_reward_over_N_episodes):
    #     x = avg_reward_over_N_episodes
    #     return 1/(1+0.5**(-x))
    #
    # def boltzman_diff(self, a, b, boltz_tau):
    #     k_t_r_n = math.exp(a / boltz_tau) - math.exp(b / boltz_tau)
    #     k_t_r_d = math.exp(a / boltz_tau) + math.exp(b / boltz_tau)
    #     k_t_r = math.fabs(k_t_r_n / k_t_r_d)
    #     return k_t_r
    #
    # def vdbe_epsilon(self, rewards_dict):
    #     rewards_dict["mt"] = (rewards_dict["mt"]/self.eps_vdbe_mt_lt_mt_tau) + rewards_dict["now"]
    #     rewards_dict["lt"] = (rewards_dict["lt"]/self.eps_vdbe_mt_lt_lt_tau) + rewards_dict["mt"]
    #     #delta_r = math.fabs(rewards_dict["lt"] - rewards_dict["mt"])
    #     k_t_r_n = math.exp(rewards_dict["mt"]/self.eps_vdbe_mt_lt_boltz_tau) - math.exp(rewards_dict["lt"]/self.eps_vdbe_mt_lt_boltz_tau)
    #     k_t_r_d = math.exp(rewards_dict["mt"]/self.eps_vdbe_mt_lt_boltz_tau) + math.exp(rewards_dict["lt"]/self.eps_vdbe_mt_lt_boltz_tau)
    #     k_t_r = math.fabs(k_t_r_n/k_t_r_d)
    #     self.eps_vdbe_mt_lt = self.eps_vdbe_mt_lt_teta*k_t_r +(1-self.eps_vdbe_mt_lt_teta)*self.eps_vdbe_mt_lt
    #
    # def scale_between_two_values(self, x, rmin, rmax, tmin, tmax):
    #     return ((x-rmin)/(rmax-rmin))*(tmax - tmin) + tmin

    def get_eps_greedy_action(self, observation):
        # Generate random number
        random_num = random.random()
        # Generate random action
        random_action = torch.from_numpy(np.random.uniform(self.as_low,self.as_high, (1, self.as_shape[0])))
        # Get greedy action
        #greedy_action = self.actor_model(observation)
        self.actor_model.eval()
        with torch.no_grad():
            greedy_action = self.actor_model(observation)

        return random_action if random_num <= self.eps else greedy_action

    def get_random_action(self, observation):
        # Generate random action
        random_action = torch.from_numpy(np.random.uniform(self.as_low,self.as_high, (1, self.as_shape[0])))
        return random_action

    def get_constant_noise_action(self, observation):
        # Generate gaussian noise
        gaussian_noise = torch.normal(mean=0, std=self.gnoisestd, size=(1,self.as_shape[0])).to(self.device)
        # Get greedy action
        self.actor_model.eval()
        with torch.no_grad():
            greedy_action = self.actor_model(observation)
        # Add noise
        noisy_action = gaussian_noise + greedy_action
        # Clamp action
        noisy_action = torch.clip(noisy_action, self.as_low_tensor, self.as_high_tensor)
        return noisy_action

    def get_constant_noise_action_eps_dependant(self, observation):
        # Generate gaussian noise
        gaussian_noise = torch.normal(mean=0, std=self.gnoisestd, size=(1, self.as_shape[0])).to(self.device)
        # Get greedy action
        self.actor_model.eval()
        with torch.no_grad():
            greedy_action = self.actor_model(observation)
        # Add noise
        noisy_action = gaussian_noise*self.eps + greedy_action
        # Clamp action
        noisy_action = torch.clip(noisy_action, self.as_low_tensor, self.as_high_tensor)
        return noisy_action

    def get_action(self, observation, rewards_dict):
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        # Default strategy is greedy
        #action = self.actor_model(observation)

        # If a strategy is selected
        if self.exp_exp_strategy_name == "just_gnoise":
            action = self.get_constant_noise_action(observation)
        elif self.exp_exp_strategy_name == "gnoise_eps-decay":
            action = self.get_constant_noise_action_eps_dependant(observation)
        elif self.exp_exp_strategy_name == "eps_greedy":
            action = self.get_eps_greedy_action(observation)
        elif self.exp_exp_strategy_name == "eps_greedy_eps-decay":
            action = self.get_eps_greedy_action(observation)
        elif self.exp_exp_strategy_name == "random":
            action = self.get_random_action(observation)
        else:
            self.actor_model.eval()
            with torch.no_grad():
                action = self.actor_model(observation)

        action = action.detach().cpu().data.numpy()[0]
        return action

    def get_eval_action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(observation)
        action = action.detach().cpu().data.numpy()[0]
        return action

    def get_data_to_train(self):
        # Sample random minibatch of transitions from memory
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        observations = torch.from_numpy(np.vstack([e[0] for e in minibatch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in minibatch if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in minibatch if e is not None])).float().to(self.device)
        next_observations = torch.from_numpy(np.vstack([e[3] for e in minibatch if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in minibatch if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return observations, actions, rewards, next_observations, dones


    def train(self, done):
        if (len(self.replay_memory) < self.minibatch_size):
            return

        observations, actions, rewards, next_observations, dones = self.get_data_to_train()

        # Train critic network
        critic_qs = self.critic_model(observations, actions)
        with torch.no_grad():
            next_actions = self.actor_target_model(next_observations)
            critic_target_model_next_qs = self.critic_target_model(next_observations, next_actions.detach())
            critic_target = rewards + self.discount_rate * critic_target_model_next_qs*(1-dones)
        #critic_loss = nn.MSELoss(critic_qs, critic_target).to(self.device)
        critic_loss = self.critic_loss(critic_qs, critic_target)

        self.critic_model.train()
        self.critic_model.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_model.optimizer.step()

        # Train actor network
        self.critic_model.eval()
        self.actor_model.optimizer.zero_grad()
        actor_loss = -self.critic_model.forward(observations, self.actor_model(observations)).mean()
        self.actor_model.train()
        actor_loss.backward()
        self.actor_model.optimizer.step()

        # Update target models parameters
        self.update_targets_params()

        # Decrement epsilon
        if self.eps > self.eps_min and done is True and "eps-decay" in self.exp_exp_strategy_name:
            self.eps -= self.eps_dec

    def save_models_checkpoints_to_disk(self):
        # Save actor and critic network
        self.actor_model.save_checkpoint()
        self.critic_model.save_checkpoint()
        # Save target actor critic networks
        self.actor_target_model.save_checkpoint()
        self.critic_target_model.save_checkpoint()
            
    def update_targets_params(self,):
        actor_model_params = self.actor_model.named_parameters()
        critic_model_params = self.critic_model.named_parameters()
        actor_target_model_params = self.actor_target_model.named_parameters()
        critic_target_model_params = self.critic_target_model.named_parameters()

        actor_model_state_dict = dict(actor_model_params)
        critic_model_state_dict = dict(critic_model_params)
        actor_target_model_state_dict = dict(actor_target_model_params)
        critic_target_model_state_dict = dict(critic_target_model_params)

        for name in critic_model_state_dict:
            critic_model_state_dict[name] = self.tau*critic_model_state_dict[name].clone() + \
                                            (1-self.tau)*critic_target_model_state_dict[name].clone()
        self.critic_target_model.load_state_dict(critic_model_state_dict)

        for name in actor_model_state_dict:
            actor_model_state_dict[name] = self.tau*actor_model_state_dict[name].clone() + \
                                            (1-self.tau)*actor_target_model_state_dict[name].clone()
        self.actor_target_model.load_state_dict(actor_model_state_dict)

        # Old code
        # # Perform target networks soft weights update
        # for target_param, param in zip(self.actor_target_model.parameters(), self.actor_model.parameters()):
        #     target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        #
        # for target_param, param in zip(self.critic_target_model.parameters(), self.critic_model.parameters()):
        #     target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
