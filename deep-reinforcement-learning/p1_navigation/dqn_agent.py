import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from dueling_model import DuelingQNetwork
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # batch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
LR_DECAY = 0.9999       # multiplicative factor of learning rate decay
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn=False, dueling_network=False, prioritized_replay=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_dqn (bool): use Double DQN method
            dueling_network (bool): use Dueling Network
            prioritized_replay (bool): use Prioritized Replay Buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dueling_network = dueling_network
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay

        random.seed(seed)

        # Q-Network
        hidden_layers = [128, 32]

        if self.dueling_network:
            hidden_state_value = [64, 32]

            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, hidden_layers, hidden_state_value).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, hidden_layers, hidden_state_value).to(device)
            self.qnetwork_target.eval()
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
            self.qnetwork_target.eval()

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, LR_DECAY)

        # Replay memory
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device, alpha=0.6, beta=0.4, beta_scheduler=1.0)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def load(self, filepath):
        # load weights from file  
        state_dict = torch.load(filepath)
        self.qnetwork_local.load_state_dict(state_dict)
        self.qnetwork_local.eval()

    def save(self, filepath):
        # Save weights to file
        torch.save(self.qnetwork_local.state_dict(), filepath)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() >= eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy()).astype(int)

        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, w) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, w = experiences

        with torch.no_grad():
            # Use of Double DQN method
            if self.double_dqn:
                # Select the greedy actions (maximum Q target for next states) from local model
                greedy_actions = self.qnetwork_local(next_states).max(dim=1, keepdim=True)[1]

                # Get the Q targets (for next states) for the greedy actions from target model
                q_targets_next = self.qnetwork_target(next_states).gather(1, greedy_actions)

            # Use of Fixed Q-Target
            else:
                # Get max predicted Q values (for next states) from target model
                q_targets_next = self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0]

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)  # shape: [batch_size, 1]

        # Compute loss
        if self.prioritized_replay:
            q_targets.sub_(q_expected)
            q_targets.squeeze_()
            q_targets.pow_(2)

            with torch.no_grad():
                td_error = q_targets
                td_error.pow_(0.5)
                self.memory.update_priorities(td_error)

            q_targets.mul_(w)
            loss = q_targets.mean()
        else:
            loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
