
import os
import logging

import tensorflow as tf

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from tsim import gameSim

class CustomEnvironment(Environment):
    gsim = gameSim()
    def __init__(self):
        super().__init__()
    
    def states(self):
        return dict(type='float', shape=(10,))

    def actions(self):
        return {"hit": dict(type="float", min_value=0.0, max_value=1.0),
                 "stay": dict(type="float", min_value=0.0, max_value=1.0)}

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional
    def close(self):
        super().close()

    def reset(self):
        self.gsim.new_hand()
        return self.gsim.state()

    def execute(self, actions):
        terminal = False
        if actions['hit'] < actions['stay']:
            self.gsim.run_dealer()
            terminal=True
        else:
            self.gsim.hit(True)
            if self.gsim.bust(True):
                terminal = True
        reward = self.gsim.reward()
        return self.gsim.state(), terminal, reward
