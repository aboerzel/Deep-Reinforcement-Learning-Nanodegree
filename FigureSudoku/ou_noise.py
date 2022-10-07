"""
To implement better exploration by the Actor network, we use noisy perturbations, specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper:
https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
It samples noise from a correlated normal distribution.
"""
import copy
import random

import numpy as np


class OUActionNoise:
    def __init__(self, mu, sigma, theta=0.15):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.state = None
        self.reset()

    def __call__(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        time_noise = self.theta * (self.mu - x)
        random_noise = self.sigma * np.random.standard_normal(self.sigma.shape[0])
        dx = time_noise + random_noise
        self.state = x + dx
        # print(f'ou-noise: {self.state} - time-noise: {time_noise} - random-noise: {random_noise}')
        return self.state

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
