import numpy as np
from utility import generate_learning_plot
import gym
from Agent import Agent
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 1500
    checkpoint_load = False

    # define agent's network parameters
    agent = Agent(gamma=0.99, epsilon=1.0, learning_rate=5e-4, input_dimensions=[8], n_actions=4, memory_size=1000000,
                  min_epsilon=0.01, size_of_batch=64, decrement_epsilon=1e-5, replace=1000)

    if checkpoint_load:
        agent.models_load()

    # to save performance plot
    fname = 'LunarLanderDDDQNPerformancePlot.png'
    scores = []
    epsilon_store = []

    for i in range(n_games):
        flag_done = False
        observe = env.reset()
        score = 0

        while not flag_done:
            action = agent.action_choice(observe)
            observe_new, reward, flag_done, information = env.step(action)
            score += reward
            agent.transition_store(observe, action, reward, observe_new, int(flag_done))
            agent.learn()
            observe = observe_new
        scores.append(score)
        score_average = np.mean(scores[-100:])
        print('Episode: ', i, 'Score %.1f' % score,
              'Average Score %.1f' % score_average,
              'Epsilon %.2f' % agent.epsilon)

        if i > 20 and i % 20 == 0:
            agent.models_save()

        epsilon_store.append(agent.epsilon)

    x_axis = [i+1 for i in range(n_games)]
    generate_learning_plot(x_axis, scores, epsilon_store, fname, 5)

