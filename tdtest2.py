import numpy as np
import gymnasium as gym
import random

def main():

    # Create taxi environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # Initializing the Q-table
    numStates = env.observation_space.n
    numActions = env.action_space.n
    qtable = np.zeros((numStates, numActions))

    # Setting a discount rate (gamma), learning rate, and epsilon
    learningRate = 0.9
    discountRate = 0.8
    epsilon = 1.0
    decayRate = 0.005

    # The model will train for 1000 episodes with 99 steps in each episode
    numEpisodes = 1000
    maxSteps = 99

    for episode in range(numEpisodes):

        # Reset the environment for each episode
        state = [env.reset()]
        done = False

        for step in range(maxSteps):
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(qtable[state[0][0], :])  # Exploit

            # Take action and observe next state and reward
            result = env.step(action)  
            #next state that happens after an action is taken
            next_state = result[0]
            #the reward for reaching this new state
            reward = result[1]
            #checks whether the state has reached the ending or not
            done = result[2]
            #extra info that comes from the step functions
            info = result[3]
            
            print("state:", state[0])
            print("action", action)
            print("next state:", next_state)

            # TD(0) update
            q_current = qtable[state[0][0], action]
            q_next_max = np.max(qtable[next_state, :])
            td_target = reward + discountRate * q_next_max
            td_error = td_target - q_current
            qtable[state[0][0], action] += learningRate * td_error

            state[0] = [next_state]

            if done == True:
                break
            
        # Decay epsilon
        epsilon = np.exp(-decayRate * episode)

    print("TD Learning completed using", numEpisodes, "episodes")
    input("Press Enter to view trained agent")

    # Test the trained agent
    state = [env.reset()]
    done = False
    totalRewards = 0

    for step in range(maxSteps):
        print("step:" ,step )
        
        
        action = np.argmax(qtable[state[0][0], :])
        result = env.step(action)  
        next_state = result[0]
        reward = result[1]
        done = result[2]
        info = result[3]
        
        totalRewards += reward
        state[0] = [next_state]

        env.render()
        print("Score:", totalRewards)

        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()
