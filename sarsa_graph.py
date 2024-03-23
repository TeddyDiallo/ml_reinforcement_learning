import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt


def main():
    # create taxi environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # initializing the q-table
    numStates = env.observation_space.n
    numActions = env.action_space.n
    qtable = np.zeros((numStates, numActions))

    # setting a discount rate (gamma), learning rate, and epsilon
    learningRate = 0.9
    discountRate = 0.8
    epsilon = 1.0
    decayRate = 0.005

    # the model will train for 1000 episodes with 99 steps in each episode
    numEpisodes = 1000
    maxSteps = 99
    numRuns = 10  # Number of runs to average over

    averaged_episode_rewards = np.zeros(numEpisodes)  # Initialize an array to store averaged rewards per episode

    for run in range(numRuns):
        episode_rewards = []  # Initialize a list to store the total rewards per episode for this run
        for episode in range(numEpisodes):
            # reset the environment each episode
            state = [env.reset()]
            done = False
            totalRewards = 0
            for s in range(maxSteps):
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    action = np.argmax(qtable[state[0][0], :])  # Exploit

                result = env.step(action)
                newState = result[0]
                reward = result[1]
                done = result[2]
                info = result[3]
                
                nextAction = np.argmax(qtable[newState,:])

                qtable[state[0][0],action] = qtable[state[0][0],action] + learningRate * (reward + discountRate * qtable[newState,nextAction]-qtable[state[0][0],action])

                state[0] = [newState]
                totalRewards += reward

                if done == True:
                    break

            epsilon = np.exp(-decayRate * episode)
            episode_rewards.append(totalRewards)

        # Aggregate the rewards per episode across all runs
        averaged_episode_rewards += np.array(episode_rewards)

    # Average the rewards per episode over all runs
    averaged_episode_rewards /= numRuns

    # Plot the averaged rewards per episode
    plt.figure(figsize=(10, 6))
    plt.plot(averaged_episode_rewards, label='Averaged Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Averaged SARSA Performance: Total Rewards per Episode')
    plt.legend()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
