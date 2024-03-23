import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

def main():
    # Create taxi environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # Initializing the Q-table and V-table
    numStates = env.observation_space.n
    numActions = env.action_space.n

    # Setting a discount rate (gamma), learning rate, and epsilon
    learningRate = 0.9
    discountRate = 0.8
    epsilon = 1.0
    decayRate = 0.005

    # The model will train for 1000 episodes with 99 steps in each episode
    numEpisodes = 1000
    maxSteps = 99
    numRuns = 30  # Number of runs to average over

    averaged_episode_rewards = np.zeros(numEpisodes)  # Initialize an array to store averaged rewards per episode

    for run in range(numRuns):
        episode_rewards = []  # Initialize a list to store the total rewards per episode for this run
        qtable = np.zeros((numStates, numActions))  # Initialize Q-table for this run
        vtable = np.zeros(numStates)  # Initialize V-table for this run
        for episode in range(numEpisodes):
            # Reset the environment for each episode
            state = [env.reset()]
            done = False
            totalRewards = 0
            for step in range(maxSteps):
                # Epsilon-greedy action selection
                if random.uniform(0, 1) < epsilon:
                    # explores a random action
                    action = env.action_space.sample()
                else:
                    # exploits the best action
                    action = np.argmax(qtable[state[0][0], :])

                # takes in action and observes new_state, reward, if done is T/F, other info that comes from the step() function
                # had to unpack the tuple that was returned from the step() function. (this gave us many issues)
                result = env.step(action)
                # next state that happens after an action is taken
                next_state = result[0]
                # the reward for reaching this new state
                reward = result[1]
                # checks whether the state has reached the ending or not
                done = result[2]
                # extra info that comes from the step functions
                info = result[3]

                # TD(0) update
                q_current = qtable[state[0][0], action]
                q_next_max = np.max(qtable[next_state, :])
                td_target = reward + discountRate * q_next_max
                td_error = td_target - q_current
                qtable[state[0][0], action] += learningRate * td_error

                # update the V-table
                td_target_v = reward + discountRate * vtable[next_state]
                td_error_v = td_target_v - vtable[state[0][0]]
                vtable[state[0][0]] += learningRate * td_error_v

                # update new state
                state[0] = [next_state]

                totalRewards += reward
                # will end once passenger is dropped off
                if done == True:
                    break

            # Decay rate using epsilon, so it explores less and exploits more towards the end
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
    plt.title('Averaged TD-Learning Performance: Total Rewards per Episode')
    plt.legend()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
