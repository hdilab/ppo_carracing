import numpy as np
from hyperparameters import *
from scipy.signal import lfilter


class Buffer:
    def __init__(self, buffer_size):
        self.states = np.zeros([buffer_size, 64, 64, 4], dtype=np.float32)
        self.actions = np.zeros([buffer_size, 3], dtype=np.float32)
        self.rewards = np.zeros([buffer_size, 1], dtype=np.float32)
        self.advantages = np.zeros([buffer_size, 1], dtype=np.float32)
        self.vals = np.zeros([buffer_size, 1], dtype=np.float32)
        self.returns = np.zeros([buffer_size, 1], dtype=np.float32)
        self.old_log_probs = np.zeros([buffer_size, 1], dtype=np.float32)

        # Pointer maintains the currently added record.
        self.ptr = 0

        # Start Path maintains the last episode start Path
        self.start_path = 0

    def store(self, state, action, reward, vals, old_log_prob):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.vals[self.ptr] = vals
        self.old_log_probs[self.ptr] = old_log_prob

        self.ptr += 1

    # While episode is finished calculate the advantages and reward to go.
    def finish(self, last_val=0):
        path_slice = slice(self.start_path, self.ptr)
        rews = np.append(self.rewards[path_slice], last_val)
        vals = np.append(self.vals[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + GAMMA * vals[1:] - vals[:-1]
        self.advantages[path_slice] = np.expand_dims(self.discount_cumsum(deltas, GAMMA * LAMBDA), 1)

        # the next line computes rewards-to-go, to be targets for the value function
        self.returns[path_slice] = np.expand_dims(self.discount_cumsum(rews, GAMMA)[:-1], 1)

        self.start_path = self.ptr

    def get(self):
        # Reset the pointer and start path
        self.ptr, self.start_path = 0, 0
        mean = np.mean(self.advantages)
        std = np.std(self.advantages)
        self.advantages = (self.advantages - mean) / (std + 1e-8)
        return self.states, self.actions, self.returns, self.advantages, self.old_log_probs

    def discount_cumsum(self, x, discount):
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
