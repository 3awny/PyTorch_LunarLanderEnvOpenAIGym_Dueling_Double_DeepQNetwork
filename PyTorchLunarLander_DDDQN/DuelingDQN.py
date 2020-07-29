import torch as torch
import torch.nn as nn
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDQN(nn.Module):
    def __init__(self, learning_rate, n_actions, name, input_dimensions, mdl_checkpoint):
        super(DuelingDQN, self).__init__()
        self.mdl_checkpoint = mdl_checkpoint
        self.file_checkpoint = os.path.join(self.mdl_checkpoint, name)

        # input layer
        self.fc1 = nn.Linear(*input_dimensions, 512)

        # dueling portion
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        # handling the device selection as this code will submitted a job to the university GPU cluster
        self.device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
        # sending the entire network to that device
        self.to(self.device)

    def forward(self, state):
        # passing state through the network and activating it with relu function
        flat1 = F.relu(self.fc1(state))
        # we then take that flattened input and get the value and advantage functions out
        value = self.value(flat1)
        advantage = self.advantage(flat1)

        return value, advantage

    def checkpoint_save(self):
        print('... checkpoint is being saved ...')
        # saving the networks state dictionary at checkpoint
        torch.save(self.state_dict(), self.file_checkpoint)

    def checkpoint_load(self):
        print('... checkpoint is being loaded ...')
        self.load_state_dict(torch.load(self.file_checkpoint))

