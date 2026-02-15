from collections import deque
import random
import torch
import numpy as np

class GridReplayBuffer:
    def __init__(self, capacity = 50_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        return_value = (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
        return return_value

    def __len__(self): return len(self.buffer)