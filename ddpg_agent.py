import math
from collections import deque
import random
import numpy as np
import torch
from torch import nn

from ddpg_networks import ActorNetwork, CriticNetwork
import os.path
from ddpg_utils import OUNoise, AdaptiveParamNoiseSpec

class DDPGAgent():
    def __init__(self, env, actor_model_path_filename, actor_target_model_path_filename, critic_model_path_filename, critic_target_model_path_filename, replay_memory,
                 exp_exp_strategy_name="", load_models_from_disk=False, minibatch_size=64, discount_rate=0.99, tau=1e-2,
                 actor_learning_rate=1e-04, critic_learning_rate=1e-03, eps_init=1.0, eps_min=0.01, eps_dec=0.0009,
                 memory_size=100000, gnoisestd=0.1, ounMU=0.0, ounTheta=0.15, ounSigma=0.3, ounDT=1e-2, apnDesiredActionStddev=.2, apnInitialStddev=.1, apnAdaptationCoefficient=1.01, apnUpdateRate=50):

        # Type of noise strategy
        self.exp_exp_strategy_name = exp_exp_strategy_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Environment related parameters
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

        # Epsilon related noise parameters
        self.eps_dec = eps_dec
        self.eps = eps_init
        self.eps_min = eps_min

        # Gaussian noise related params
        self.gnoisestd = gnoisestd

        # OUNoise
        self.ounoise = OUNoise(self.as_shape, ounMU, ounTheta, ounSigma, ounDT)

        # Adaptive parameter noise params
        self.apnDistances = []
        self.apnDesiredActionStddev = apnDesiredActionStddev
        self.apnInitialStddev = apnInitialStddev
        self.apnAdaptationCoefficient = apnAdaptationCoefficient
        self.apnNoise = AdaptiveParamNoiseSpec(self.apnInitialStddev, self.apnDesiredActionStddev, self.apnAdaptationCoefficient)
        self.apnUpdateRate = apnUpdateRate

        # Replay memory parameters
        self.memory_size = memory_size
        self.replay_memory = deque(maxlen=self.memory_size) if replay_memory is None else replay_memory
        self.minibatch_size = minibatch_size

        self.load_models_from_disk = load_models_from_disk

        # Use if we need to collect history of models loss
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.actor_target_loss_history = []
        self.critic_target_loss_history = []

        # Models related parameters
        self.actor_model_path_filename = actor_model_path_filename
        self.actor_target_model_path_filename = actor_target_model_path_filename
        self.critic_model_path_filename = critic_model_path_filename
        self.critic_target_model_path_filename = critic_target_model_path_filename
        self.discount_rate = discount_rate
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau
        # Original params 400 300
        # self.actor_fc1_pnum = 200
        # self.actor_fc2_pnum = 150
        # self.critic_fc1_pnum = 200
        # self.critic_fc2_pnum = 150

        self.actor_fc1_pnum = 400
        self.actor_fc2_pnum = 300
        self.critic_fc1_pnum = 400
        self.critic_fc2_pnum = 300

        self.build_models()

    def build_models(self):

        self.actor_model = ActorNetwork(self.actor_learning_rate, self.os_shape[0], self.actor_fc1_pnum, self.actor_fc2_pnum, self.as_shape[0], self.actor_model_path_filename, 'ActorNetwork')
        self.actor_target_model = ActorNetwork(self.actor_learning_rate, self.os_shape[0], self.actor_fc1_pnum, self.actor_fc2_pnum, self.as_shape[0], self.actor_target_model_path_filename, 'TargetActorNetwork')

        self.critic_model = CriticNetwork(self.critic_learning_rate, self.os_shape[0], self.critic_fc1_pnum, self.critic_fc2_pnum, self.as_shape[0], self.critic_model_path_filename, 'CriticNetwork')
        self.critic_target_model = CriticNetwork(self.critic_learning_rate, self.os_shape[0], self.critic_fc1_pnum, self.critic_fc2_pnum, self.as_shape[0], self.critic_target_model_path_filename, 'TargetCriticNetwork')

        # Noised Actor for Adaptive Parametric Noise
        # This will never be saved
        self.noisy_actor_model = ActorNetwork(self.actor_learning_rate, self.os_shape[0], self.actor_fc1_pnum, self.actor_fc2_pnum, self.as_shape[0], '', 'NoisyActorNetwork')
        self.perturb_actor_parameters()
        #self.noisy_actor_model.load_state_dict(self.actor_model.state_dict().copy())
        #self.noisy_actor_model.add_parameter_noise(self.apnNoise.get_current_stddev())

        if self.load_models_from_disk and os.path.isfile(self.critic_model_path_filename) and os.path.isfile(self.critic_target_model_path_filename) and os.path.isfile(self.actor_model_path_filename) and os.path.isfile(self.actor_target_model_path_filename):
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

    def get_eps_greedy_action(self, observation):
        # Generate random number
        random_num = random.random()
        # Generate random action
        random_action = torch.from_numpy(np.random.uniform(self.as_low,self.as_high, (1, self.as_shape[0]))).to(self.device)
        # Get greedy action
        #greedy_action = self.actor_model(observation)
        self.actor_model.eval()
        with torch.no_grad():
            greedy_action = self.actor_model(observation)

        return random_action if random_num <= self.eps else greedy_action

    def get_random_action(self, observation):
        # Generate random action
        random_action = torch.from_numpy(np.random.uniform(self.as_low,self.as_high, (1, self.as_shape[0]))).to(self.device)
        random_action = torch.clip(random_action, self.as_low_tensor, self.as_high_tensor)
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

    def get_constant_ounoise(self, observation):
        #Get OUNoise
        ounoise = self.ounoise.get_noise()
        # Get greedy action
        self.actor_model.eval()
        with torch.no_grad():
            greedy_action = self.actor_model(observation)
            # Add noise
            noisy_action = torch.from_numpy(ounoise).to(self.device) + greedy_action
            # Clamp action
            noisy_action = torch.clip(noisy_action, self.as_low_tensor, self.as_high_tensor)
            return noisy_action

    def ddpg_distance_metric(self, actions1, actions2):
        """
        Compute "distance" between actions taken by two policies at the same states
        Expects numpy arrays
        """
        diff = actions1 - actions2
        s = np.square(diff)
        dist = math.sqrt(np.mean(s))
        #mean_diff = np.mean(s, axis=0)
        #dist = math.sqrt(np.mean(mean_diff))
        return dist

    def get_adaptive_parametric_noise(self, observation):
        self.noisy_actor_model.eval()
        #self.actor_model.eval()
        with torch.no_grad():
            # Get greedy action
            #greedy_action = self.actor_model(observation)
            # Hard copy the actor model params to noisy actor model
            #self.noisy_actor_model.load_state_dict(self.actor_model.state_dict().copy())
            # Add noise to the params of noisy actor model
            #self.noisy_actor_model.add_parameter_noise(self.apnNoise.get_current_stddev())
            # Get noisy action from noisy actor model
            noisy_action = self.noisy_actor_model(observation)
            # Clamp action
            noisy_action = torch.clip(noisy_action, self.as_low_tensor, self.as_high_tensor)

            return noisy_action


    def get_action(self, observation, rewards_dict=[]):
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
        elif self.exp_exp_strategy_name == "ounoise":
            action = self.get_constant_ounoise(observation)
        elif self.exp_exp_strategy_name == "adaptive-parameter-noise":
            action = self.get_adaptive_parametric_noise(observation)
        elif self.exp_exp_strategy_name == "no-noise":
            action = self.get_greedy_action(observation)
        elif self.exp_exp_strategy_name == "no-noise-without-layer-normalization":
            action = self.get_greedy_action(observation)
        else:
            # In any other case get greedy action
            self.actor_model.eval()
            with torch.no_grad():
                action = self.actor_model(observation)

        action = action.detach().cpu().data.numpy()[0]
        return action

    def get_eval_mod_action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        action = self.get_greedy_action(observation)
        action = action.detach().cpu().data.numpy()[0]
        return action

    def get_eval_action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(observation)
        action = action.detach().cpu().data.numpy()[0]
        return action

    def get_greedy_action(self, observation):
        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(observation)
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

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def perturb_actor_parameters(self):
        """Apply parameter noise to actor model, for exploration"""
        self.hard_update(self.noisy_actor_model, self.actor_model)
        params = self.noisy_actor_model.state_dict()
        for name in params:
            if 'ln' in name:
                continue
            param = params[name].to(self.device)
            param += torch.randn(param.shape).to(self.device) * self.apnNoise.get_current_stddev()

    def train(self, steps, done):
        if (len(self.replay_memory) < self.minibatch_size):
            return

        observations, actions, rewards, next_observations, dones = self.get_data_to_train()

        # Update Adaptive Parameters Noise params
        if self.exp_exp_strategy_name == "adaptive-parameter-noise" and (steps%self.apnUpdateRate==0):
            #self.noisy_actor_model.load_state_dict(self.actor_model.state_dict().copy())
            #self.hard_update(self.noisy_actor_model, self.actor_model)
            #self.noisy_actor_model.add_parameter_noise(self.apnNoise.get_current_stddev())
            self.perturb_actor_parameters()
            self.actor_model.eval()
            self.noisy_actor_model.eval()
            with torch.no_grad():
                unperturbed_actions = self.actor_model(observations)
                perturbed_actions = self.noisy_actor_model(observations)

            ddpg_dist = self.ddpg_distance_metric(unperturbed_actions.detach().cpu().data.numpy(), perturbed_actions.detach().cpu().data.numpy())
            self.apnNoise.adapt(ddpg_dist)

        #Train without traget networks using CrossNorm from paper "CrossNorm: On Normalization for Off-Policy TD Reinforcement Learning"
        # next_actions = self.actor_model(next_observations)
        # obs_next_obs = torch.cat((observations, next_observations), 0)
        # acts_next_acts = torch.cat((actions, next_actions.detach()), 0)
        # xonoff = self.critic_model(obs_next_obs, acts_next_acts)

        # with torch.no_grad():
        #     self.critic_model.eval()
        #     self.actor_model.eval()
        #
        #     xoff = self.critic_model(observations, actions)
        #     xon = self.critic_model(next_observations, next_actions)
        #     xcoff = xoff.mean().detach().cpu().data.item()
        #     xcon = xon.mean().detach().cpu().data.item()
        #     mucalfa = 0.5*xcoff + (1-0.5)*xcon
        #     var = (math.pow(xcon-mucalfa,2)+math.pow(xcoff-mucalfa,2))/((self.minibatch_size*2)-1)
        #
        #     q_target = (xonoff - mucalfa)/(math.sqrt(var+0.00001))
        #
        # critic_loss = self.critic_loss(xonoff, q_target)

        #Train critic network
        self.critic_model.eval()
        critic_qs = self.critic_model(observations, actions)
        with torch.no_grad():
            if "without-target-networks" in self.exp_exp_strategy_name:
                next_actions = self.actor_model(next_observations)
                critic_model_next_qs = self.critic_model(next_observations, next_actions.detach())
                critic_target = rewards + self.discount_rate * critic_model_next_qs * (1 - dones)
            else:
                next_actions = self.actor_target_model(next_observations)
                critic_target_model_next_qs = self.critic_target_model(next_observations, next_actions.detach())
                critic_target = rewards + self.discount_rate * critic_target_model_next_qs*(1-dones)

        self.critic_model.train()
        critic_loss = self.critic_loss(critic_qs, critic_target)
        self.critic_model.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_model.optimizer.step()

        # Train actor network
        self.critic_model.eval()
        self.actor_model.optimizer.zero_grad()
        actor_loss = -self.critic_model(observations, self.actor_model(observations)).mean()
        self.actor_model.train()
        actor_loss.backward()
        self.actor_model.optimizer.step()

        # Update target models parameters
        if "without-target-networks" not in self.exp_exp_strategy_name:
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
