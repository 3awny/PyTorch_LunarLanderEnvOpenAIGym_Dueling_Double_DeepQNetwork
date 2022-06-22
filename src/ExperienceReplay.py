import torch as T
import torch.nn as nn
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ExperienceReplay:
    def __init__(self, maximum_size, inp_shape):
        self.memory_size = maximum_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *inp_shape), dtype=np.float32)
        self.state_memory_new = np.zeros((self.memory_size, *inp_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.memory_size, dtype=np.uint8)

    # storing state-action transitions
    def transition_store(self, state, action, reward, new_state, flag_done):
        idx = self.memory_counter % self.memory_size
        self.state_memory[idx] = state
        self.state_memory_new[idx] = new_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = flag_done
        self.memory_counter += 1

    def buffer_sample(self, size_of_batch):
        maximum_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(maximum_memory, size_of_batch, replace=False)
        states = self.state_memory[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        new_states = self.state_memory_new[batch]
        terminal = self.terminal_mem[batch]

        return states, actions, rewards, new_states, terminal




