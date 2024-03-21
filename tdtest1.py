import numpy as np
import gymnasium as gym
import random


def main():
    # Create taxi environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # Initializing the value function (V-table)
    numStates = env.observation_space.n
    V = np.zeros(numStates)

    # Setting learning rate, discount rate, and epsilon
    learningRate = 0.9
    discountRate = 0.8
    epsilon = 1.0
    decayRate = 0.001

    # Training parameters
    numEpisodes = 1000
    maxSteps = 99

    for episode in range(numEpisodes):
        state = env.reset()
        done = False

        for step in range(maxSteps):
            # Use epsilon-greedy policy to choose action
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax([sum([p * (reward + discountRate * V[next_state]) for p, next_state, reward, _ in env.P[state[0]][action]]) for action in range(env.action_space.n)])

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
            # Update value function using TD learning
            target = reward + discountRate * V[next_state]
            delta = target - V[state[0]]
            V[state[0]] += learningRate * delta

            if done:
                break
            
            state = [state]
            state[0] = next_state

        # Decay epsilon
        epsilon = max(0.01, epsilon - decayRate)

    # Training completed
    print("TD Learning training completed using", numEpisodes, "episodes")
    input("Press Enter to view trained agent")

    # Watch the trained agent
    state = env.reset()
    done = False
    totalRewards = 0

    for step in range(maxSteps):
        print("step:" ,step )
        action = np.argmax([sum([p * (reward + discountRate * V[next_state]) for p, next_state, reward, _ in env.unwrapped.P[state[0]][action]]) for action in range(env.action_space.n)])
        
        # Take action and observe next state and reward
        result = env.step(action)  
        next_state = result[0]
        reward = result[1]
        done = result[2]
        info = result[3]

        totalRewards += reward

        env.render()
        print("Score:", totalRewards)

        if done:
            break

        state = [state]
        state[0] = next_state

    env.close()

if __name__ == "__main__":
    main()
