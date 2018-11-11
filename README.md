# drlnd-nav
Navigation Project as part of the Deep Reinforcement Learning Nanodegree at Udacity

# Introduction
The project aims to solve an environment similar to the Banana Collection Unity environment employing Value-Based Methods for Deep Reinforcement Learning.

# Getting Started
Follow this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies) to setup the Udacity DRLND conda enviroment.

There are 2 ways to explore the code:
1. By following the guided iPython notebook - _Navigation.ipynb_
2. By directly running _navigation.py_ from command line
   * Command Line argument to be passed : _train_ for training and _test_ for testing
   
       `($) python navigation.py train`
   
       `($) python navigation.py test`
   
The enivorments for this project are placed in the _env_ folder. Select the environment as per your OS, unzip the file and update the path to the in the respecting file (_Navigation.ipynb_ or _navigation.py_)    

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
