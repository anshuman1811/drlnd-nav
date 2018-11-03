from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import argparse, sys

# Add argument for train/test mode
parser=argparse.ArgumentParser()
parser.add_argument('mode', help='train if training else test', type=str)
args=parser.parse_args()

print ("Mode = ", args.mode)

env = UnityEnvironment(file_name="C:/Users/anshmish/Desktop/Personal/Courses/DRLND/deep-reinforcement-learning/p1_navigation/Banana_Windows_x86_64/Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

def trainAgent(agent, n_episodes=2000, timeout=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, score_window_size=100, solution_score=13.0):
    print('\nTraining agent ')
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=score_window_size)  # last 'score_window_size' scores for candidate solution
    eps = eps_start                    # initialize epsilon
    
    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(timeout):
            # Query agent for action
            action = agent.act(state)
            # Get feedback from environment
            env_info = env.step(action.item())[brain_name]             # send the action to the environment
            next_state = env_info.vector_observations[0]           # get the next state
            reward = env_info.rewards[0]                           # get the reward
            done = env_info.local_done[0]                          # see if episode has finished

            # Move the agent a step
            agent.step(state, action, reward, next_state, done)

            score += reward                                    # update the score
            state = next_state                                 # roll over the state to next time step
            if done:                                           # exit loop if episode finished
                break
        
        # Cache the scores and decay epsilon
        scores.append(score)
        scores_window.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        # Print episode results
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")

        # Print if solution score achieved
        if np.mean(scores_window)>=solution_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    print("\nTraining Completed!")
    return scores
    
def testAgent():
    print("Testing the Agent")
    agent = Agent(state_size=state_size, action_size=action_size, seed=0, pretrainedWeightsFile='checkpoint.pth', train = False)
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state)                    # select an action
        env_info = env.step(action.item())[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    print("Score: {}".format(score))
    return score

def plotScores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if args.mode == 'train':
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    scores = trainAgent (agent)
    plotScores(scores)
elif args.mode == 'test':
    testAgent()
else:
	print("Invalid Mode")

