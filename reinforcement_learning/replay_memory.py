
import numpy as np
import random

from state_generator import K_SIZE

INIT_SIZE = 2**14
DEFAULT_CAPACITY = 2**24 
BATCH_SIZE = 2**12

class ReplayMemory(object):

    def __init__(self, batch_size=BATCH_SIZE, init_size=INIT_SIZE, capacity=DEFAULT_CAPACITY):
        self.memory = {}
        self.batch_size = batch_size
        self.first_index = -1
        self.last_index = -1
        self.capacity = capacity
        self.init_size = init_size

    def store(self, sars_tuple):
        if self.first_index == -1:
            self.first_index = 0
        self.last_index += 1
        self.memory[self.last_index] = sars_tuple
        if (self.last_index + 1 - self.first_index) > self.capacity:
            self.discard_sample()

    def canTrain(self):
        return self.last_index + 1 - self.first_index >= self.init_size

    def is_full(self):
        return self.last_index + 1 - self.first_index >= self.capacity

    def is_empty(self):
        return self.first_index == -1

    def discard_sample(self):
        rand_index = self.first_index
        del self.memory[rand_index]
        self.first_index += 1

    def sample(self):
        if self.is_empty():
            raise Exception('Unable to sample from replay memory when empty')
        rand_sample_index = random.randint(self.first_index, self.last_index)
        return self.memory[rand_sample_index]

    def sample_batch(self):
        # must insert data into replay memory before sampling
        if not self.canTrain():
            return (-1, -1, -1, -1)
        if self.is_empty():
            raise Exception('Unable to sample from replay memory when empty')

        # determine shape of states
        states_shape = (self.batch_size,) + np.shape(self.memory.values()[0][0])
        # + np.shape(self.memory.values()[0][2])
        rewards_shape = (self.batch_size * K_SIZE)
        nextStates_shape = (self.batch_size * K_SIZE, 5)

        states = np.empty(states_shape)
        actions = np.empty((self.batch_size, 1))
        rewards = np.empty(rewards_shape)
        next_states = np.empty(nextStates_shape)

        # sample batch_size times from the memory
        for idx in range(self.batch_size):
            state, action, reward, next_state, = self.sample()
            states[idx] = state
            actions[idx] = action
            rewards[idx * K_SIZE:((idx + 1) * K_SIZE)] = reward
            next_states[idx * K_SIZE:((idx + 1) * K_SIZE)] = next_state

        return (states, actions, rewards, next_states)
