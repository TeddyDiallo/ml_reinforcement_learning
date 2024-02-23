import numpy as np
import gymnasium as gym
import random

def main():

    # create Taxi environment
    env = gym.make('Taxi-v3',render_mode="human")

    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    num_episodes = 10
    max_steps = 5 

    for episode in range(num_episodes):

        # reset the environment
        state = [env.reset()]
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state[0][0],:])

            # take action and observe reward
            result = env.step(action)  
            new_state = result[0]
            reward = result[1]
            done = result[2]
            info = result[3]
            
            print("State:", state[0][0])
            print("Action:", action)

    
            # Q-learning algorithm
            qtable[state[0][0],action] = qtable[state[0][0],action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state[0][0],action])

            # Update to our new state
            print("new state:", new_state)
            state[0] = [new_state]


            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = [env.reset()]
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state[0][0],:])
        result = env.step(action)  
        new_state = result[0]
        reward = result[1]
        done = result[2]
        info = result[3]
    
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state[0] = [new_state]

        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()