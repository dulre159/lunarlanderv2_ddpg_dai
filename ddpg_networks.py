import numpy as np
import torch as T
from torch import nn, optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, critic_lr, input_size, fc1_size, fc2_size, actions_size, chkpt_file='', model_name=''):
        super(CriticNetwork, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.actions_size = actions_size
        self.checkpoint_file = chkpt_file
        self.lr = critic_lr
        self.model_name = model_name

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.ln1 = nn.LayerNorm(self.fc1_size)

        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        f2 = 2. / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.ln2 = nn.LayerNorm(self.fc2_size)

        self.action_q = nn.Linear(self.actions_size, self.fc2_size)
        # f3 is -3e-3 from DDPG paper
        f3 = 0.003
        self.q = nn.Linear(self.fc2_size, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.ln3 = nn.LayerNorm(self.fc2_size)

        self.ln4 = nn.LayerNorm(self.fc2_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_q = self.fc1(state)
        # In paper "CrossNorm: On Normalization for Off-Policy TD
        # Reinforcement Learning" they state that "in the case of LayerNorm where applying
        # normalization to the input layer produces worse results."
        # state_q = self.ln1(state_q)
        state_q = F.relu(state_q)
        # BN after activation?
        # According to: "https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md" it gives better accuracy
        state_q = self.ln1(state_q)
        state_q = self.fc2(state_q)
        # BN after activation?
        state_q = F.relu(state_q)
        state_q = self.ln2(state_q)

        action_q = self.action_q(action)
        action_q = F.relu(action_q)
        # BN on actions added for try
        action_q = self.ln3(action_q)

        state_action_q = T.add(state_q, action_q)
        state_action_q = F.relu(state_action_q)
        # BN after activation?
        state_action_q = self.ln4(state_action_q)

        state_action_q = self.q(state_action_q)

        return state_action_q

    def save_checkpoint(self):
        print(self.model_name+': saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(self.model_name+': loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, actor_lr, input_size, fc1_size, fc2_size,
                 out_size, chkpt_file='', model_name=''):
        super(ActorNetwork, self).__init__()

        self.lr = actor_lr
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.out_size = out_size
        self.checkpoint_file = chkpt_file
        self.model_name = model_name

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -self.f1, self.f1)
        T.nn.init.uniform_(self.fc1.bias.data, -self.f1, self.f1)
        
        self.ln1 = nn.LayerNorm(self.fc1_size)

        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.f2 = 2. / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -self.f2, self.f2)
        T.nn.init.uniform_(self.fc2.bias.data, -self.f2, self.f2)

        self.ln2 = nn.LayerNorm(self.fc2_size)

        self.policy_value = nn.Linear(self.fc2_size, self.out_size)
        # f3 is -3e-3 from DDPG paper
        self.f3 = 0.003
        T.nn.init.uniform_(self.policy_value.weight.data, -self.f3, self.f3)
        T.nn.init.uniform_(self.policy_value.bias.data, -self.f3, self.f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    #
    # def add_parameter_noise(self, scalar=.1):
    #         # self.fc1.weight.data += T.randn_like(self.fc1.weight.data) * scalar
    #         # self.fc2.weight.data += T.randn_like(self.fc2.weight.data) * scalar
    #         # self.policy_value.weight.data += T.randn_like(self.policy_value.weight.data) * scalar
    #
    #
    #         self.fc1.weight.data +=  T.normal(mean=0, std=scalar, size=tuple(self.fc1.weight.data.shape)).to(self.device)
    #         self.fc1.bias.data += T.normal(mean=0, std=scalar, size=tuple(self.fc1.bias.data.shape)).to(self.device)
    #         self.fc2.weight.data += T.normal(mean=0, std=scalar, size=tuple(self.fc2.weight.data.shape)).to(self.device)
    #         self.fc2.bias.data += T.normal(mean=0, std=scalar, size=tuple(self.fc2.bias.data.shape)).to(self.device)
    #         self.policy_value.weight.data += T.normal(mean=0, std=scalar, size=tuple(self.policy_value.weight.data.shape)).to(self.device)
    #         self.policy_value.bias.data += T.normal(mean=0, std=scalar, size=tuple(self.policy_value.bias.data.shape)).to(self.device)
    #
    #         # self.fc1.weight.data += T.randn_like(self.fc1.weight.data) * scalar
    #         # self.fc1.bias.data += T.randn_like(self.fc1.bias.data) * scalar
    #         # self.fc2.weight.data += T.randn_like(self.fc2.weight.data) * scalar
    #         # self.fc2.bias.data += T.randn_like(self.fc2.bias.data) * scalar
    #         # self.policy_value.weight.data += T.randn_like(self.policy_value.weight.data) * scalar
    #         # self.policy_value.bias.data += T.randn_like(self.policy_value.bias.data) * scalar
    #
    #         # fc1WData = self.fc1.weight.data.detach().clone()
    #         # fc1BData = self.fc1.bias.data.detach().clone()
    #         # T.nn.init.uniform_(fc1WData, -self.f1, self.f1)
    #         # T.nn.init.uniform_(fc1BData, -self.f1, self.f1)
    #         # self.fc1.weight.data += fc1WData * scalar
    #         # self.fc1.bias.data += fc1BData * scalar
    #         #
    #         # fc2WData = self.fc2.weight.data.detach().clone()
    #         # fc2BData = self.fc2.bias.data.detach().clone()
    #         # T.nn.init.uniform_(fc2WData, -self.f2, self.f2)
    #         # T.nn.init.uniform_(fc2BData, -self.f2, self.f2)
    #         # self.fc2.weight.data += fc2WData * scalar
    #         # self.fc2.bias.data += fc2BData * scalar
    #         #
    #         # policy_valueWData = self.policy_value.weight.data.detach().clone()
    #         # policy_valueBData = self.policy_value.bias.data.detach().clone()
    #         # T.nn.init.uniform_(policy_valueWData, -self.f3, self.f3)
    #         # T.nn.init.uniform_(policy_valueBData, -self.f3, self.f3)
    #         # self.policy_value.weight.data += policy_valueWData * scalar
    #         # self.policy_value.bias.data += policy_valueBData * scalar


    def forward(self, state):
        action = self.fc1(state)
        # In paper "CrossNorm: On Normalization for Off-Policy TD
        # Reinforcement Learning" they state that "in the case of LayerNorm where applying
        # normalization to the input layer produces worse results."
        # action = self.ln1(action)
        action = F.relu(action)
        # BN after activation?
        # According to: "https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md" it gives better accuracy
        action = self.ln1(action)
        action = self.fc2(action)
        # action = self.ln2(action)
        action = F.relu(action)
        # BN after activation?
        action = self.ln2(action)
        action = T.tanh(self.policy_value(action))
        return action

    # def forward_noisy(self, state, perturbation_scale=.1):
    #     self.add_parameter_noise(perturbation_scale)
    #
    #     action = self.fc1(state)
    #     # In paper "CrossNorm: On Normalization for Off-Policy TD
    #     # Reinforcement Learning" they state that "in the case of LayerNorm where applying
    #     # normalization to the input layer produces worse results."
    #     # action = self.ln1(action)
    #     action = F.relu(action)
    #     # BN after activation?
    #     # According to: "https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md" it gives better accuracy
    #     action = self.ln1(action)
    #     action = self.fc2(action)
    #     # action = self.ln2(action)
    #     action = F.relu(action)
    #     # BN after activation?
    #     action = self.ln2(action)
    #     action = T.tanh(self.policy_value(action))
    #     return action

    def save_checkpoint(self):
        print(self.model_name+': saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(self.model_name+': loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))