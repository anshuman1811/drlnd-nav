# drlnd-nav
Navigation Project as part of the Deep Reinforcement Learning Nanodegree at Udacity

# Introduction
The project aims to solve an environment similar to the Banana Collection Unity environment employing Value-Based Methods for Deep Reinforcement Learning. The objective is to train an agent which can navigate a large square world collecting yellow bananas while avoiding blue ones.

# Getting Started
Follow this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies) to setup the Udacity DRLND conda enviroment.

There are 2 ways to explore the code:
1. By following the guided iPython notebook - _Navigation.ipynb_
2. By directly running _navigation.py_ from command line
   * Command Line argument to be passed : _train_ for training and _test_ for testing
   
       `($) python navigation.py train`
   
       `($) python navigation.py test`
   
Download the environment for this project as per your OS from the following links:
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Linux Headless](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Update the path to the environment in the respecting file (_Navigation.ipynb_ or _navigation.py_)    

- **Mac**: `path/to/Banana.app`,
- **Windows** (x86): `path/to/Banana_Windows_x86/Banana.exe`,
- **Windows** (x86_64): `path/to/Banana_Windows_x86_64/Banana.exe`,
- **Linux** (x86): `path/to/Banana_Linux/Banana.x86`,
- **Linux** (x86_64): `path/to/Banana_Linux/Banana.x86_64`,
- **Linux** (x86, headless): `path/to/Banana_Linux_NoVis/Banana.x86`,
- **Linux** (x86_64, headless): `path/to/Banana_Linux_NoVis/Banana.x86_64`,

Update the following files to tweak the model or the agent:
- `model.py` defines the Neural Network
- `dqn_agent.py`defines the behavior of the DQN Agent

# The Environment
## State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

## Action Space
Four discrete actions are available, corresponding to:
```
0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
```

## Rewards
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

## Solution Criteria
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
