import numpy
import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt

def plot_learning_curve(x, total_rewards_per_episode_history, plot_filename):
    running_avg = np.zeros(len(total_rewards_per_episode_history) - (1 if len(total_rewards_per_episode_history) > len(x) else 0))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(total_rewards_per_episode_history[max(0, i-100):(i+1)])
    if len(running_avg) == len(x) and len(x)+len(running_avg)> 0:
        fig1, ax1 = plt.subplots()
        ax1.plot(x, running_avg, color='blue', marker='o', linestyle='-')
        ax1.set_title('Moving average of rewards of 100 per episode')
        ax1.set_xlabel('train episodes')
        ax1.set_ylabel('mean reward over prev 100 eps')
        fig1.savefig(plot_filename)
        plt.close(fig1)
        # plt.plot(x, running_avg, 'o')
        # plt.title('Moving average of rewards of 100 per episode')
        # plt.xlabel('train episodes')
        # plt.xlabel('mean reward')
        # plt.savefig(plot_filename)


def plot_eval_curve(eval_ep, tot_ep, tot_eval_ep_avg_reward_history, plot_filename):
    x = []
    for nn in range(0, tot_ep+2):
        if nn % eval_ep == 0:
            x.append(nn)
        #if nn == tot_ep - 1:
            #x.append(nn + 1)
    if len(tot_eval_ep_avg_reward_history) == len(x) and len(x)+len(tot_eval_ep_avg_reward_history) > 0:
        # plt.plot(x, tot_eval_ep_avg_reward_history, 'x')
        # plt.xlabel("train episodes")
        # plt.ylabel("mean reward")
        # plt.savefig(plot_filename)

        fig2, ax2 = plt.subplots()
        ax2.plot(x, tot_eval_ep_avg_reward_history,color='orange', marker='x', linestyle='-')
        ax2.set_xlabel('train episodes')
        ax2.set_ylabel('mean reward over 50 eval ep')
        fig2.savefig(plot_filename)
        plt.close(fig2)

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_shape, mu=0.0, theta=0.15, sigma=0.3, dt=1e-2, x0=None):
        self._action_dim = action_shape
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._x0 = x0 if x0 is not None else self._mu * np.zeros(self._action_dim)
        self._state = self._x0

    def reset(self):
        """
        Reset the OU process.
        Should be done after each episode
        """
        self._state = self._x0

    def evolve_state(self):
        """
        Advance the OU process.
        Returns: np.ndarray: Updated OU process state.
        """
        x = self._state
        dx = self._theta * (self._mu - x) * self._dt + self._sigma * np.sqrt(
            self._dt) * np.random.normal(size=len(x))
        self._state = x + dx
        return self._state

    def get_noise(self):
        ou_state = self.evolve_state()
        return ou_state

"""
From OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev
        self.distHistory = np.zeros((0), dtype=float)
        self.distSum = 0
        self.distNum = 0

    def adapt(self, distance):
        self.distSum += distance
        self.distNum += 1
        self.distHistory = numpy.append(self.distHistory, distance)
        print("DDPG ANP DISTANCE RESULT:"+str(distance))
        #print("DDPG ANP DESIRED DISTANCE:"+str(self.desired_action_stddev))
        #print("DDPG ANP DESIRED DISTANCE MEAN:"+str(self.distSum/self.distNum))
        if distance > self.desired_action_stddev:
        # if distance > (self.distSum/self.distNum):
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_current_stddev(self):
        return self.current_stddev
