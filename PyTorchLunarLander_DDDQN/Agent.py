import torch as torch
import numpy as np
from ExperienceReplay import ExperienceReplay
from DuelingDQN import DuelingDQN


class Agent:
    def __init__(self, gamma, epsilon, learning_rate, n_actions, input_dimensions, memory_size, size_of_batch, \
                 min_epsilon=0.01, decrement_epsilon=5e-7, replace=1000, mdl_checkpoint='temp/duelingDDQN'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dimensions = input_dimensions
        self.size_of_batch = size_of_batch
        self.min_epsilon = min_epsilon
        self.decrement_epsilon = decrement_epsilon
        self.cnt_target_replace = replace
        self.mdl_checkpoint = mdl_checkpoint
        self.action_space = [i for i in range(self.n_actions)]
        self.memory = ExperienceReplay(memory_size, input_dimensions)
        self.counter_learn = 0

        # instance of the deep q-network - tell value of the current state
        self.q_network_eval = DuelingDQN(self.learning_rate, self.n_actions, input_dimensions=self.input_dimensions,
                                         name='lunarLanderDuelingDDQN_q_network_eval',
                                         mdl_checkpoint=self.mdl_checkpoint)
        # instance of the deep q-network - tell value of the next actions
        self.q_network_next = DuelingDQN(self.learning_rate, self.n_actions, input_dimensions=self.input_dimensions,
                                         name='lunarLanderDuelingDDQN_q_network_next',
                                         mdl_checkpoint=self.mdl_checkpoint)

    # choosing action
    def action_choice(self, observe):
        # exploitation
        if np.random.random() > self.epsilon:
            # convert observation to state tensor (PyTorch tensor), send it to our device and feed-forward
            # it through our network and get the advantage function out
            state = torch.tensor([observe], dtype=torch.float).to(self.q_network_eval.device)
            # we get the advantage function out. We don't care about the value function
            # as it is just a constant; does not affect anything hence _ is placed
            _, advntge = self.q_network_eval.forward(state)
            # the advantage function is used to calculate the maximal action
            # the argmax function returns a PyTorch tensor, which the OpenAI Gym will not accept as
            # input for its step function, so a numpy array is passed with the .item() function
            action = torch.argmax(advntge).item()
        # exploration
        else:
            action = np.random.choice(self.action_space)  # random choice from the action space
        return action

    # storing state-action transitions (interface with agent's memory)
    def transition_store(self, state, action, reward, new_state, flag_done):
        self.memory.transition_store(state, action, reward, new_state, flag_done)

    def target_network_replace(self):
        # counter_learn is how many times the agent executed the learning function
        if self.counter_learn % self.cnt_target_replace == 0:
            # loading the state dictionary from the evaluation network onto the Q next network
            self.q_network_next.load_state_dict(self.q_network_eval.state_dict())

    # linear epsilon decay
    def epsilon_decay(self):
        self.epsilon = self.epsilon - self.decrement_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon

    # agent's saving network functionality
    def models_save(self):
        self.q_network_eval.checkpoint_save()
        self.q_network_next.checkpoint_save()

    # agent's loading network functionality
    def models_load(self):
        self.q_network_eval.checkpoint_load()
        self.q_network_next.checkpoint_load()

    # learning functionality
    def learn(self):
        # addressing if the agent hasn't filled up enough memory to preform learning
        # e.g. size of batch = 64 memory samples in each learning step. Lets say the agent
        # has so far only completed 10 steps or even 1 step. So there is not enough memory
        # yet to satisfy the set size of batch. We handle this by waiting until the agent fills
        # up its memory to the size of batch.
        if self.memory.memory_counter < self.size_of_batch:
            return

        # in pytorch the first thing you want to in a learning function is zeroing the gradience on the optimizer
        self.q_network_eval.optimizer.zero_grad()

        self.target_network_replace()

        # sampling of memory
        state, action, reward, next_state, flag_done = self.memory.buffer_sample(self.size_of_batch)

        # converting numpy arrays to pytorch tensors
        states = torch.tensor(state).to(self.q_network_eval.device)
        actions = torch.tensor(action).to(self.q_network_eval.device)
        flag_dones = torch.tensor(flag_done).to(self.q_network_eval.device)
        rewards = torch.tensor(reward).to(self.q_network_eval.device)
        next_states = torch.tensor(next_state).to(self.q_network_eval.device)

        # array from 0 to size_of_batch-1 that handles array indexing and slicing later on
        indices = np.arange(self.size_of_batch)

        # passing in states and next states to the respective networks
        value_s, advantage_s = self.q_network_eval.forward(states)
        value_s_new, advantage_s_new = self.q_network_next.forward(next_states)
        # this line comes from the methodology of the paper introducing Double Deep Q-learning
        value_s_eval, advantage_s_eval = self.q_network_eval.forward(next_states)
        # the former three quantities pairs are needed to perform the update rule
        # based on the paper introducing Double Deep Q-learning

        # dueling aggregation of value and advantage functions
        # In the paper introducing Dueling Deep Q-learning they settle on summing the value and advantage function
        # with normalizing by subtracting off the mean of the advantage stream. Summing them alone without this
        # normalization step will lead a problem called "identifiability", which is discussed in the report.

        # the array indexing on the line below, takes the indices of the size_of_batch as an array of indices and the
        # values of the actions the agent actually took by taking the actions sub-array
        q_network_pred = torch.add(value_s, (advantage_s - advantage_s.mean(dim=1, keepdim=True)))[indices, actions]
        # we perform the indexing below as we want it for all actions
        q_network_next = torch.add(value_s_new, (advantage_s_new - advantage_s_new.mean(dim=1, keepdim=True)))
        q_network_eval = torch.add(value_s_eval, (advantage_s_eval - advantage_s_eval.mean(dim=1, keepdim=True)))

        # maximal actions of the next state according to the evaluation network
        maximum_actions = torch.argmax(q_network_eval, dim=1)

        # evaluating rewards for which the next state is terminal
        # does not value future states that are flagged as terminal
        q_network_next[flag_dones] = 0.0
        # quantity of the target value is q_network_next according to the evaluation network
        q_network_target = rewards + self.gamma * q_network_next[indices, maximum_actions]

        # calculation of the loss function
        loss = self.q_network_eval.loss_func(q_network_target, q_network_pred).to(self.q_network_eval.device)
        # back-propagation
        loss.backward()
        # stepping the optimiser
        self.q_network_eval.optimizer.step()
        # increment learn function counter
        self.counter_learn += 1
        # epsilon decay
        self.epsilon_decay()