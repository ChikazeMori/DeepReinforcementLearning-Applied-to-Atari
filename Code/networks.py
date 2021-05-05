import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        # CNN
        self.conv1 = nn.Conv1d(self.input_dims[0]+n_actions, 256, 1, stride=4)
        self.conv2 = nn.Conv1d(256, 256, 1, stride=2)
        self.conv3 = nn.Conv1d(256, 256, 1, stride=1)
        self.pool = nn.MaxPool1d(1, 1)
        
        # fully connected layers
        self.fc1 = nn.Linear(self.fc1_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        
        # CNN
        x = self.conv1(T.cat([state, T.reshape(action,(256,1))], dim=1).unsqueeze(2))
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1_dims)
        
        # fully connected layers
        action_value = self.fc1(x)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
    
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        # CNN
        self.conv1 = nn.Conv1d(*self.input_dims, 256, 1, stride=4)
        self.conv2 = nn.Conv1d(256, 256, 1, stride=2)
        self.conv3 = nn.Conv1d(256, 256, 1, stride=1)
        self.pool = nn.MaxPool1d(1, 1)
        
        # fully connected layers
        self.fc1 = nn.Linear(self.fc1_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        
        # CNN
        x = self.conv1(state.unsqueeze(2))
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1_dims)
        
        # fully connected layers
        state_value = self.fc1(x)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=1, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6       
        
        # CNN
        self.conv1 = nn.Conv1d(*self.input_dims, 256, 1, stride=4)
        self.conv2 = nn.Conv1d(256, 256, 1, stride=2)
        self.conv3 = nn.Conv1d(256, 256, 1, stride=1)
        self.pool = nn.MaxPool1d(1, 1)         
        
        # fully connected layers
        self.fc1 = nn.Linear(self.fc1_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, 9)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        
        # CNN
        x = self.conv1(state.unsqueeze(2))
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1_dims)
        
        # fully connected layers
        prob = self.fc1(x)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return sigma

    def sample_normal(self, state):
        sigma = self.forward(state)
        
        c_probabilities = Categorical(sigma)
        c_actions = c_probabilities.sample()
        c_action = c_actions*T.tensor(self.max_action).to(self.device)
        c_log_probs = c_probabilities.log_prob(c_actions)
        c_log_probs -= T.log(1-c_action.pow(2)+self.reparam_noise)

        return c_actions, c_log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))