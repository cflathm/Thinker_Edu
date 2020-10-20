# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import logging

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
import cenv
from tsim import gameSim
from tensorforce import Runner


# Create an OpenAI-Gym environment
environment = Environment.create(
    environment='cenv.CustomEnvironment', max_episode_timesteps=10
)

# Create a PPO agent
agent = Agent.create(
    agent='ppo',
    environment=environment,
    # Automatically configured network
    network='auto',
    # PPO optimization parameters
    batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
    subsampling_fraction=0.33,
    # Reward estimation
    likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
    # Baseline network and optimizer
    baseline=dict(type='auto', size=32, depth=1),
    baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
    # Regularization
    l2_regularization=0.0, entropy_regularization=0.0,
    # Preprocessing
    state_preprocessing='linear_normalization', reward_preprocessing=None,
    # Exploration
    exploration=0.0, variable_noise=0.0,
    # Default additional config values
    config=None,
    parallel_interactions=1,
    # Save agent every 10 updates and keep the 5 most recent checkpoints
    saver=dict(directory='model', frequency=10, max_checkpoints=5),
    # Log all available Tensorboard summaries
    summarizer=dict(directory='summaries', summaries='all'),
    # Do not record agent-environment interaction trace
    recorder=None
)

# Initialize the runner
runner = Runner(agent=agent, environment=environment)

# Start the runner
runner.run(num_episodes=10000)
runner.close()
agent.save(directory='.', format='numpy', append='episodes')
exit()


gsim = gameSim()
for i in range(50):
    terminal = False
    while not terminal:
        print("AI: " + str(gsim.player))
        print("Dealer: " + str(gsim.dealer))
        actions = agent.act(gsim.state())
        print(actions)
        if actions['hit'] < actions['stay']:
            gsim.run_dealer()
            terminal=True
        else:
            gsim.hit(True)
            if gsim.bust(True):
                terminal = True
        agent.observe(reward=gsim.reward(), terminal=terminal)
    print("AI: " + str(gsim.player))
    print("Dealer: " + str(gsim.dealer))
    print("================\n=============")
    gsim.new_hand()