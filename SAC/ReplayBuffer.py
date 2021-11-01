import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_mem_size, input_shape, n_actions):
        self.max_mem_size = max_mem_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.mem_count = 0

        self.state_memory = np.zeros((self.max_mem_size, *self.input_shape))
        self.next_state_memory = np.zeros((self.max_mem_size, *self.input_shape))
        self.action_memory = np.zeros((self.max_mem_size, self.n_actions))
        self.reward_memory = np.zeros(self.max_mem_size)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_count % self.max_mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.max_mem_size, self.mem_count)

        batch = np.random.choice(max_mem, batch_size)

        b_state = self.state_memory[batch]
        b_action = self.action_memory[batch]
        b_reward = self.reward_memory[batch]
        b_state_ = self.next_state_memory[batch]
        b_terminal = self.terminal_memory[batch]

        return b_state, b_action, b_reward, b_state_, b_terminal
