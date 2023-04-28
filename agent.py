import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from collections import deque


class ReplayBuffer:
    '''
    saves transition datas to buffer
    '''

    def __init__(self, buffer_size=100000, n_step=1, gamma=0.85):
        '''
        Replay Buffer initialize function

        args:
            buffer_size: maximum size of buffer
            n_step: n step if using n step DQN
            gamma: discount factor for n step
        '''
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

        self.n_states = deque(maxlen=self.n_step)
        self.n_actions = deque(maxlen=self.n_step)
        self.n_rewards = deque(maxlen=self.n_step)
        self.n_next_states = deque(maxlen=self.n_step)
        self.n_dones = deque(maxlen=self.n_step)


    def __len__(self) -> int:
        return len(self.states)


    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        
        if self.n_step > 1:
            self.n_states.append(state)
            self.n_actions.append(action)
            self.n_rewards.append(reward)
            self.n_next_states.append(next_state)
            self.n_dones.append(done)
            
            if len(self.n_states) == self.n_step:
                # append to main buffer by preprocessing n step
                self.states.append(self.n_states[0])
                self.actions.append(self.n_actions[0])
                reward_sum = 0
                for i, reward in enumerate(self.n_rewards):
                    reward_sum += self.gamma ** i * reward
                self.rewards.append(reward_sum)
                self.next_states.append(next_state)
                self.dones.append(self.n_dones[-1])
            if done:
                self.n_states.clear()
                self.n_actions.clear()
                self.n_rewards.clear()
                self.n_next_states.clear()
                self.n_dones.clear()
                
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

    
    def sample(self, batch_size, device=None):
        '''
        samples random batches from buffer

        args:
            batch_size: size of the minibatch
            device: pytorch device

        returns:
            states, actions, rewards, next_states, dones
        '''
        idx = random.sample(range(len(self.states)),batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in idx:
            states.append(self.states[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            next_states.append(self.next_states[i])
            dones.append(self.dones[i])

        return torch.tensor(states,dtype=torch.float), torch.tensor(actions), torch.tensor(rewards), torch.tensor(next_states,dtype=torch.float), torch.tensor(dones)

class DQN(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''
    def __init__(self, input_size, output_size):
        '''
        Define your architecture here
        '''
        super().__init__()
        hidden_size = 64
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        #self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        #nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, state):
        '''
        Get Q values for each action given state
        '''
        q_values = F.relu(self.layer1(state))
        q_values = F.relu(self.layer2(q_values))
        #q_values = F.relu(self.layer3(q_values))
        q_values = self.layer4(q_values)

        return q_values


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.curr_step = 0

        self.learning_rate = 0.0001
        self.buffer_size = 50000
        self.batch_size = 64
        self.epsilon = 0.10
        self.gamma = 0.995
        self.n_step = 1
        self.target_update_freq = 512
        self.gradient_update_freq = 1
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = DQN(state_size, action_size).to(self.device)
        self.target_network = deepcopy(self.network)

        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)


    def select_action(self, state, is_test=False):
        '''
        selects action given state

        returns:
            discrete action integer
        '''
        if np.random.uniform() < self.epsilon and is_test == False:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            return np.argmax(self.network(torch.from_numpy(state)).numpy())



    def train_network(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            target = self.target_network.forward(next_states)
            target_q = target.max(1)[0]
            target_q = rewards + (self.gamma ** self.n_step) * target_q * dones

        cur = self.network.forward(states)
        q = cur.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        '''
        updates the target network to online
        '''
        # Use deepcopy of online network
        self.target_network = deepcopy(self.network)


    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size and self.curr_step % self.gradient_update_freq == 0:
            self.train_network(*self.replay_buffer.sample(self.batch_size, device=self.device))
            if((self.curr_step % self.target_update_freq) == 0):
                self.update_target_network()


