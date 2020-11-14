# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from ddpg_agent import Agent
import matplotlib.pyplot as plt
from collections import deque
import torch
import random
from IPython import get_ipython

# %% [markdown]
# # Continuous Control
#
# ---
#
# In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
#
# ### 1. Start the Environment
#
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# %%
from unityagents import UnityEnvironment
import numpy as np

# %% [markdown]
# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
#
# - **Mac**: `"path/to/Reacher.app"`
# - **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
# - **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
# - **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
# - **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
# - **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
# - **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`
#
# For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Reacher.app")
# ```

# %%
env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

# %% [markdown]
# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# %%
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# %% [markdown]
# ### 2. Examine the State and Action Spaces
#
# In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
#
# The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.
#
# Run the code cell below to print some information about the environment.

# %%
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(
    states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# %% [markdown]
# ### 3. Take Random Actions in the Environment
#
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
#
# Once this cell is executed, you will watch the agent's performance, if it selects an action at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.
#
# Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!

# %%
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
# get the current state (for each agent)
states = env_info.vector_observations
# initialize the score (for each agent)
scores = np.zeros(num_agents)
while True:
    # select an action (for each agent)
    actions = np.random.randn(num_agents, action_size)
    # all actions between -1 and 1
    actions = np.clip(actions, -1, 1)
    # send all actions to tne environment
    env_info = env.step(actions)[brain_name]
    # get next state (for each agent)
    next_states = env_info.vector_observations
    # get reward (for each agent)
    rewards = env_info.rewards
    dones = env_info.local_done                        # see if episode finished
    # update the score (for each agent)
    scores += env_info.rewards
    # roll over states to next time step
    states = next_states
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

# %% [markdown]
# ### 4. It's Your Turn!
#
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# %% [markdown]
# ## Imports

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Create the Agents

# %%
agents = [Agent(state_size=state_size, action_size=action_size,
                random_seed=i) for i in range(num_agents)]

# %% [markdown]
# ## Train the Agent with DDPG

# %%


def ddpg(warm_up=int(1e4), n_episodes=300, max_t=10000, print_every=100):
    all_scores_deques = [deque(maxlen=print_every) for i in range(num_agents)]
    all_scores = np.zeros((num_agents, 0))
    total_steps = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        for agent in agents:
            agent.reset()
        scores = np.zeros(num_agents)
        for t in range(max_t):
            if total_steps < warm_up:
                # select an action (for each agent)
                actions = np.random.randn(num_agents, action_size)
                # all actions between -1 and 1
                actions = np.clip(actions, -1, 1)
            else:
                actions = [agent.act(state)
                           for agent, state in zip(agents, states)]

            total_steps += 1
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for agent, state, action, reward, next_state, done in zip(agents, states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        for score, scores_deque in zip(scores, all_scores_deques):
            scores_deque.append(score)
        all_scores = np.append(all_scores, np.reshape(
            scores, (num_agents, 1)), axis=1)

        avg_score = np.mean(all_scores_deques)
        high_score = np.max(scores)
        high_scorer = agents[np.argmax(scores)]

        print('\rEpisode {}\tAverage Score (over agents): {:.2f}\tScore: {:.2f}'.format(
            i_episode, avg_score, high_score), end="")
        torch.save(high_scorer.actor_local.state_dict(),
                   'checkpoint_actor.pth')
        torch.save(high_scorer.critic_local.state_dict(),
                   'checkpoint_critic.pth')

        if avg_score >= 30.0:
            print('\rSolved at episide {}!  With average score: {:.2f}\tScore: {:.2f}').format(
                i_episode, avg_score, high_score)

    return all_scores


all_scores = ddpg(n_episodes=100)

fig, axs = plt.subplots(num_agents)
for ax, scores in zip(axs.flat if num_agents > 1 else [axs], all_scores):
    ax.plot(np.arange(1, len(scores)+1), scores)
    ax.set(xlabel='Episode #', ylabel='Score')
    ax.plot()

# %% [markdown]
# When finished, you can close the environment.

# %%
env.close()


# %%
