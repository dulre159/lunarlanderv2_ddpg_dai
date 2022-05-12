import numpy as np
import torch as T
from torch import nn, optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, critic_lr, input_size, fc1_size, fc2_size, actions_size, chkpt_file=''):
        super(CriticNetwork, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.actions_size = actions_size
        self.checkpoint_file = chkpt_file
        self.lr = critic_lr

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_size)

        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        f2 = 2. / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_size)

        self.action_q = nn.Linear(self.actions_size, self.fc2_size)
        # f3 is -3e-3 from DDPG paper
        f3 = 0.003
        self.q = nn.Linear(self.fc2_size, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_q = self.fc1(state)
        state_q = self.bn1(state_q)
        state_q = F.relu(state_q)
        state_q = self.fc2(state_q)
        state_q = self.bn2(state_q)

        action_q = self.action_q(action)
        action_q = F.relu(action_q)
        state_action_q = T.add(state_q, action_q)
        state_action_q = F.relu(state_action_q)
        state_action_q = self.q(state_action_q)

        return state_action_q

    def save_checkpoint(self):
        print('CriticNetwork: saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('CriticNetwork: loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, actor_lr, input_size, fc1_size, fc2_size,
                 out_size, chkpt_file=''):
        super(ActorNetwork, self).__init__()

        self.lr = actor_lr
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.out_size = out_size
        self.checkpoint_file = chkpt_file

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        self.bn1 = nn.LayerNorm(self.fc1_size)

        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        f2 = 2. / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_size)

        self.policy_value = nn.Linear(self.fc2_size, self.out_size)
        # f3 is -3e-3 from DDPG paper
        f3 = 0.003
        T.nn.init.uniform_(self.policy_value.weight.data, -f3, f3)
        T.nn.init.uniform_(self.policy_value.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        action = self.fc1(state)
        action = self.bn1(action)
        action = F.relu(action)
        action = self.fc2(action)
        action = self.bn2(action)
        action = F.relu(action)
        action = T.tanh(self.policy_value(action))
        return action

    def save_checkpoint(self):
        print('ActorNetwork: saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('ActorNetwork: loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class NextStateRewardGeneratorModel(nn.Module):
    def __init__(self, input_size, input_layer_output_size, fc_hidden_layer_one_input_size,
                 fc_hidden_layer_one_output_size, output_layer_output_size, output_size):
        super(ActorNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_layer_output_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_layer_one_input_size, fc_hidden_layer_one_output_size),
            nn.ReLU(),
            nn.Linear(output_layer_output_size, output_size),
            nn.Tanh()
        )

        #The TanH in the end is needed to keep the outputs between -/+1

    def forward(self, x):
        actions = self.linear_relu_stack(x)
        return actions