from collections import deque
import random
import numpy as np
import gym
import torch
from torch import nn, optim
from FCQNetwork import FCQNetwork as QNetwork
import os.path

class DeepQAgent():
    def __init__(self, env, device, main_model_name, target_model_name, load_models_from_disk=True, minibatch_size=128, discount_rate=0.95, learning_rate=0.0005, eps_init=0.01, eps_min=0.01, eps_dec=0.0001, memory_size=100000, steps_to_train=100, steps_to_synch=500):
        #Observation space might be discrete while action_space must be discrete
        self.is_os_discrete = type(env.observation_space) == gym.spaces.discrete.Discrete

        self.action_size = env.action_space.n
        if self.is_os_discrete:
            self.os_size = env.observation_space.n
        else:
            self.os_shape = env.observation_space.shape

        self.main_model_name = main_model_name
        self.target_model_name = target_model_name
        self.eps_dec = eps_dec
        self.last_steps_from_train = 0
        self.last_steps_from_synch = 0
        self.running_loss = 0
        self.eps = eps_init
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.eps_min = eps_min
        self.memory_size = memory_size
        self.steps_to_train = steps_to_train
        self.steps_to_synch = steps_to_synch
        self.device = device
        self.replay_memory = deque(maxlen=self.memory_size)
        self.minibatch_size = minibatch_size
        self.load_models_from_disk = load_models_from_disk

        self.build_models()

    def build_models(self):
        if self.is_os_discrete:
            self.main_model = QNetwork(self.os_size, self.action_size).to(self.device)
            self.target_model = QNetwork(self.os_size, self.action_size).to(self.device)
        else:
            self.main_model = QNetwork(self.os_shape[0], self.action_size).to(self.device)
            self.target_model = QNetwork(self.os_shape[0], self.action_size).to(self.device)

        if self.load_models_from_disk and os.path.isfile(self.main_model_name+".pt") and os.path.isfile(self.target_model_name+".pt"):
            self.main_model = torch.load(self.main_model_name+".pt")
            self.target_model = torch.load(self.target_model_name+".pt")
        self.target_model.load_state_dict(self.main_model.state_dict())

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.learning_rate)

    def save_state(self, state):
        observation, action, reward, next_observation, done = state
        self.replay_memory.append([observation, action, reward, next_observation, done])

    def get_action(self, state):
        # Let's balance between greedy and random exploration
        # print(state)
        x = torch.from_numpy(state).to(self.device)
        self.main_model.eval()
        with torch.no_grad():
            actions_qs_list = self.main_model(x)
        # print(actions_qs_list)
        action_greedy = np.argmax(actions_qs_list.cpu().data.numpy())
        #action_greedy = torch.max(actions_qs_list).item()
        action_random = np.random.randint(0, self.action_size)
        return action_random if random.random() <= self.eps else action_greedy

    def train(self, steps, done):
        if (steps - self.last_steps_from_train < self.steps_to_train) or (len(self.replay_memory) < self.minibatch_size):
            return
        self.last_steps_from_train = steps
        # Sample random minibatch of transitions from memory
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        # print("Minibatch: {}".format(minibatch))
        observations = torch.from_numpy(np.vstack([e[0] for e in minibatch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in minibatch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in minibatch if e is not None])).float().to(self.device)
        next_observations = torch.from_numpy(np.vstack([e[3] for e in minibatch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in minibatch if e is not None]).astype(np.uint8)).float().to(
            self.device)

        self.main_model.eval()
        self.target_model.eval()

        qvalues_list_main = self.main_model(observations).gather(1, actions)
        qvalues_list_target = []
        with torch.no_grad():
            qvalues_list_target = self.target_model(next_observations).detach().max(1)[0].unsqueeze(1)

        self.main_model.train()
        yjs = rewards + (self.discount_rate*qvalues_list_target*(1-dones))
        loss = self.criterion(qvalues_list_main, yjs).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.running_loss += loss.item()

        torch.save(self.main_model, self.main_model_name+".pt")

        # Perform target network weights update
        if steps-self.last_steps_from_synch >= self.steps_to_synch:
            self.last_steps_from_synch = steps
            self.target_model.train()
            self.target_model.load_state_dict(self.main_model.state_dict())
            torch.save(self.target_model, self.target_model_name+".pt")

        # Dec epsilon
        if self.eps > self.eps_min:
            self.eps = self.eps - self.eps_dec