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
                for step in range(self.n_step):
                    self.states.append(self.n_states.popleft())
                    self.actions.append(self.n_actions.popleft())
                    self.rewards.append(self.n_rewards.popleft())
                    self.next_states.append(self.n_next_states.popleft())
                    self.dones.append(self.n_dones.popleft())
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
        ret = deque(maxlen=batch_size)

        bufStartIdx = np.random.randint(self.buffer_size - batch_size)
        self.states.rotate(bufStartIdx)
        self.actions.rotate(bufStartIdx)
        self.rewards.rotate(bufStartIdx)
        self.next_states.rotate(bufStartIdx)
        self.dones.rotate(bufStartIdx)

        states = deque(maxlen=batch_size)
        actions = deque(maxlen=batch_size)
        rewards = deque(maxlen=batch_size)
        next_states = deque(maxlen=batch_size)
        dones = deque(maxlen=batch_size)
        for i in range(batch_size):
            states.append(self.states.popleft())
            actions.append(self.actions.popleft())
            rewards.append(self.rewards.popleft())
            next_states.append(self.next_states.popleft())
            dones.append(self.dones.popleft())
        return states, actions, rewards, next_states, dones

class DQN(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''
    def __init__(self, input_size, output_size):
        '''
        Define your architecture here
        '''
        super().__init__()
        hidden_size = 256
        self.example_layer = nn.Linear(input_size, output_size)
        self.example_activation = nn.Tanh()
       

    def forward(self, state):
        '''
        Get Q values for each action given state
        '''
        q_values = self.example_layer(state)
        return q_values


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.curr_step = 0

        self.learning_rate = 0.0003
        self.buffer_size = 50000
        self.batch_size = 64
        self.epsilon = 0.10
        self.gamma = 0.85
        self.n_step = 4
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
        return np.random.randint(self.action_size)


    def train_network(self, states, actions, rewards, next_states, dones):
        y = rewards.popleft()
        accGamma = self.gamma
        for ns in range(self.n_step):
            if ns == self.n_step - 1:
                y += accGamma * np.np.max(self.target_network.forward(next_states.popleft()))
            else:
                y += accGamma * rewards.popleft()
            accGamma *= self.gamma





        loss = (y - self.network.forward(states.popleft())[actions.popleft()])^2



    def update_target_network(self):
        '''
        updates the target network to online
        '''
        # Use deepcopy of online network
        assert NotImplementedError


    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size and self.curr_step % self.gradient_update_freq == 0:
            self.train_network(*self.replay_buffer.sample(self.batch_size, device=self.device))


