# Learning Algorithm

The algorithm used for the implementation was a **Deep Q-network** based on the [DeepMind paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

An enhancement over the approach of the paper was to do a soft update of the target network parameters through an extra hyperparameter _TAU_. This adds stability to the training.

Epsilon-Greedy policy was used for action selection.

## Hyperparameters
### Agent Hyperparameters
```
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

### Training Hyperparameters
```
N_EPISODES=2000         # number of episodes to train for
TIMEOUT=1000            # timeout for each episode
EPS_START=1.0           # starting epsilon value for epsilon-greedy action selection
EPS_END=0.01            # limiting point for epsilon decay
EPS_DECAY=0.995         # epsilon decay rate
```

# Model Architecture
In contrast to the DQN paper implementation of the algorithm, the model architecture of the neural network used here consists of 3 fully connected layers (defined in _model.py_). This is because the input to the network is a state defined through 37 parameters as compared to raw pixel values.
```
FC1                 : 64 output units
FC2                 : 64 output units
FC3 (output layer)  : 4 ouput units
```
The outputs of FC1 and FC2 were fed to ReLU activation units.

# Training Results
The solution criteria used for training was an average score of 13.0 over 100 consecutive episodes.

The agent took about 1300 episodes to solve the environment.

The average training scores of the agent:
![Average Training Scores of the Agent](/Training_Avg_Scores.png)

# Ideas for Future Work
## Double DQN
Using max operators for both selection and evaluation leads to overoptimistic value estimates. To mitigate the problem, DDQN uses current network to select an action and feeds it to the target network to get an estimate. Reference : [DDQN Paper](https://arxiv.org/abs/1509.06461)

## Prioritized Experience Replay
The idea is of prioritized replay is to increase the replay probability of experience tuples that have a high expected learning progress. It uses the absolute TD error as proxy to evaluate the same. This helps with faster learning as well as a better final policy. Reference : [Prioritized Experience Replay Paper](https://arxiv.org/abs/1511.05952)

## Dueling DQN (train using pixel values)
The dueling network achitecture creates separate branches to estimate the state value function and the advantage function while sharing the common set of convolutional layers. The idea behind it is that for many states, it is unnecessary to estimate the value of each action choice. The dueling architecture combines the result of the 2 branches to output the Q function and is thus, a complementary appraoch to existing algorithms and enhancements. Reference : [Dueling Network Architecture paper](https://arxiv.org/abs/1511.06581)
