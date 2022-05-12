# import math
import os.path
import signal
import sys
import threading
import pickle
# import time
# from subprocess import PIPE, STDOUT, Popen
# import statistics
# import torch
# import torch.optim as optim
import gym
# import random
import numpy as np
# import time
# from IPython.display import clear_output
# from collections import deque
# from DQNAgent import DeepQAgent
from DDPGAgent import DDPGAgent
from DDPGUtils import  plot_learning_curve
# import keyboard
import time
# from threading import Thread
import ast

original_sigint = signal.getsignal(signal.SIGINT)

user_input = [None]

def get_user_input(user_input_ref):
    user_input_ref[0] = input("Input [S or s] to STOP - [P or p] to PAUSE\n")

def exit_gracefully(signum, frame):
    # Restore the original signal handler as otherwise evil things will happen
    # In raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)
    sys.stdout.write('S\n')
    user_input[0] = "s"
    # Restore the exit gracefully handler here
    signal.signal(signal.SIGINT, exit_gracefully)

def save_last_run_replay_memory(last_run_replay_memory_filename, ddpgAgent):
    print("Saving last run replay memory on file: " + last_run_replay_memory_filename + "...")
    try:
        last_run_replay_memory_file = open(last_run_replay_memory_filename, "wb")
        last_run_replay_memory_file.truncate(0)
        pickle.dump(ddpgAgent.replay_memory, last_run_replay_memory_file)
        last_run_replay_memory_file.close()
    except OSError:
        print("OSError: Could not write last run replay memory file...")
        return None
    except Exception as err:
        print("Unexpected error opening/writing to last run replay memory file: " + repr(err))
        return None


def load_last_run_replay_memory(last_run_replay_memory_filename):
    print("Loading last run replay memory on file: " + last_run_replay_memory_filename + "...")
    try:
        last_run_replay_memory_file = open(last_run_replay_memory_filename, "rb")
        replay_memory = pickle.load(last_run_replay_memory_file)
        last_run_replay_memory_file.close()

        return replay_memory

    except OSError:
        print("OSError: Could not write last run replay memory file...")
        return None
    except Exception as err:
        print("Unexpected error opening/writing to last run replay memory file: " + repr(err))
        return None


def save_last_run_info(last_run_filename, ep, n_eps, rewards_dict, ddpgAgent, tot_ep_reward_history, tot_eval_ep_avg_reward_history):
    print("Saving last run information on file: "+ last_run_filename +"...")
    try:
        last_run_file = open(last_run_filename, "w+")
        last_run_file.truncate(0)
        last_run_file.write(str(ep) + ";" + \
                            str(n_eps) + ";" + \
                            # str(rewards_dict["mt"]) + ";" + \
                            # str(rewards_dict["lt"]) + ";" + \
                            # str(ddpgAgent.eps_vdbe_mt_lt) + ";" + \
                            str(ddpgAgent.eps) + ";" + \
                            # str(rewards_dict["max"]) + ";" + \
                            str(tot_ep_reward_history) + ";" + \
                            str(tot_eval_ep_avg_reward_history))
        last_run_file.close()
    except OSError:
        print("OSError: Could not write last run information file...")
        return None
    except Exception as err:
        print("Unexpected error opening/writing to last run information file: " + repr(err))
        return None

def load_last_run_info(last_run_filename):
    print("Loading last run information file: "+last_run_filename+"...")
    if not os.path.exists(last_run_filename) or not os.path.exists(last_run_replay_memory_filename):
        print("Last run information file not found...")
        return None

    try:
        last_run_file = open(last_run_filename, 'r')
        file_content = last_run_file.read()
        last_run_file.close()

        if file_content.strip() == "" or file_content.count(";") != 4:
            return None

        data = file_content.split(";")

        return int(data[0]), int(data[1]), float(data[2]), ast.literal_eval(data[3]), ast.literal_eval(data[4])

    except OSError:
        print("OSError: Could not read last run information file...")
        return None
    except Exception as err:
        print("Unexpected error opening last run information file: " + repr(err))
        return None

def plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history):
    print("Plotting running mean of rewards history...")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pf = plot_filename + timestr + '.png'
    x = [i + 1 for i in range(ep)]
    plot_learning_curve(x, tot_ep_reward_history, pf)


if __name__ == '__main__':

    signal.signal(signal.SIGINT, exit_gracefully)

    input_thread = threading.Thread(target=get_user_input, args=(user_input,))
    input_thread.daemon = True
    input_thread.start()

    #exp_exp_strategy_name = "just_gnoise"
    #exp_exp_strategy_name = "gnoise_eps-decay"
    exp_exp_strategy_name = "eps_greedy"
    #exp_exp_strategy_name = "eps_greedy_eps-decay"
    #exp_exp_strategy_name = "random"

    exp_exp_strategy_filename = exp_exp_strategy_name

    load_models_from_disk = False
    load_last_run_from_disk = False
    load_last_run_replay_memory_from_disk = False

    env_name = "LunarLanderContinuous-v2"
    #env_name = "BipedalWalker-v3"
    env = gym.make(env_name)

    rewards_dict ={"avg_10_ep": 0, "avg_50_ep": 0, "avg_100_ep": 0, "avg_tot_ep":0, "avg_tot_ep_min":10000000000, "avg_tot_ep_max":-10000000000, "mt":0, "lt":0, "now": 0, "max": 0, "variance" : 0, "std" : 0}

    eps_init=0.1
    eps_min=0.001
    eps_dec=0.000001
    gnoisestd=0.1

    # Change filename depending on the selected strategy
    if exp_exp_strategy_name == "just_gnoise":
        exp_exp_strategy_filename += "_" + "std" + str(gnoisestd).rstrip(".")
    if exp_exp_strategy_name == "gnoise_eps-decay":
        exp_exp_strategy_filename += "_" + \
        "std" + str(gnoisestd).replace(".", "p") + "_" + \
        "eps-init" + str(eps_init).replace(".", "p") + "_" + \
        "eps-dec" + str(eps_dec).replace(".", "p") + "_" + \
        "eps-min" + str(eps_min).replace(".", "p")
    if exp_exp_strategy_name == "eps_greedy":
        exp_exp_strategy_filename += "_" + \
        "eps-init" + str(eps_init).replace(".", "p")
    if exp_exp_strategy_name == "eps_greedy_eps-decay":
        exp_exp_strategy_filename += "_" + \
        "eps-init" + str(eps_init).replace(".", "p") + "_" \
        "eps-dec" + str(eps_dec).replace(".", "p") + "_" + \
        "eps-min" + str(eps_min).replace(".", "p")


    #Create needed directories if they do not exist
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir('last_runs'):
        os.mkdir('last_runs')
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    plot_filename = 'plots/' + env_name + "_" + exp_exp_strategy_filename + "_"
    last_run_filename = 'last_runs/' + env_name+"_"+exp_exp_strategy_filename+"_last_run.txt"
    last_run_replay_memory_filename = 'last_runs/' + env_name+"_"+exp_exp_strategy_filename+"_last_run_replay_memory.bin"

    last_run_information = None
    last_run_replay_memory = None

    if load_last_run_from_disk:
        # Result of the below func will be None or a tuple like (ep, n_eps, epsilon, tot_ep_reward_history, tot_eval_ep_avg_reward_history)
        last_run_information = load_last_run_info(last_run_filename)

        if last_run_information is not None:
            rewards_dict["avg_10_ep"] = np.mean(last_run_information[3][-10:])
            rewards_dict["avg_100_ep"] = np.mean(last_run_information[3][-100:])
            rewards_dict["avg_tot_ep"] = np.mean(last_run_information[3][-last_run_information[0]:])
            #rewards_dict["max"] = max_reward
            eps_init = last_run_information[2]

    if load_last_run_replay_memory_from_disk:
        last_run_replay_memory = load_last_run_replay_memory(last_run_replay_memory_filename)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    ddpgAgent = DDPGAgent(
        env,
        "checkpoints/ddpg-actor_"+env_name+"_"+exp_exp_strategy_filename,
        "checkpoints/ddpg-actor-target_"+env_name+"_"+exp_exp_strategy_filename,
        "checkpoints/ddpg-critic_"+env_name+"_"+exp_exp_strategy_filename,
        "checkpoints/ddpg-critic-target_"+env_name+"_"+exp_exp_strategy_filename,
        last_run_replay_memory,
        exp_exp_strategy_name,
        load_models_from_disk,
        #eps_vdbe_mt_lt=lr_vdbe_mt_lt_eps if load_last_run_from_disk else 1,
        gnoisestd=gnoisestd,
        eps_init=eps_init,
        eps_min=eps_min,
        eps_dec=eps_dec
        )
    steps = 0
    n_eps = 1000 if not load_last_run_from_disk or last_run_information is None else last_run_information[1]
    ep = 0 if not load_last_run_from_disk or last_run_information is None else last_run_information[0]
    tot_ep_reward_history = [] if not load_last_run_from_disk or last_run_information is None else last_run_information[3]
    tot_eval_ep_avg_reward_history = [] if not load_last_run_from_disk or last_run_information is None else last_run_information[4]
    ep_reward_best = env.reward_range[0]
    init_max_reward = True if not load_last_run_from_disk else False
    save_ep = 50
    eval_ep = 50
    n_eval_eps = 50
    max_ep_steps = 100
    max_eval_ep_steps = 100
    ep_render_step = 100

    for ep in range(ep, n_eps):

        # Save step
        if ep % save_ep == 0:
            print("--- Starting Saving Phase ---")
            print("Saving models...")
            ddpgAgent.save_models_checkpoints_to_disk()

            save_last_run_replay_memory(last_run_replay_memory_filename, ddpgAgent)
            save_last_run_info(last_run_filename, ep, n_eps, rewards_dict, ddpgAgent, tot_ep_reward_history, tot_eval_ep_avg_reward_history)

            plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history)

            print("--- End Of Saving Phase ---")


        #Evaluation step
        if ep % eval_ep == 0:
            print("--- Starting Evaluation Phase ---")
            tot_eval_eps_reward = 0
            for eep in range(0, n_eval_eps):
                eval_obs = env.reset()
                eval_done = False
                total_eval_ep_reward = 0
                eval_ep_steps = 0
                while not eval_done:
                    eval_action = ddpgAgent.get_action(eval_obs, rewards_dict)
                    eval_next_observation, eval_reward, eval_done, eval_info = env.step(eval_action)
                    eval_obs = eval_next_observation
                    total_eval_ep_reward += eval_reward
                    eval_ep_steps +=1

                    # To speed up evaluation max number of steps are 400
                    if eval_ep_steps == max_eval_ep_steps:
                        eval_done = True

                    if eval_done:
                        print("Eval-EP: {}, TOT-R: {}, STEPS: {}".format(eep, total_eval_ep_reward, eval_ep_steps))
                    env.render()
                tot_eval_eps_reward += total_eval_ep_reward
            print("--- End Of Evaluation Phase ---")
            tot_eval_ep_avg_reward_history.append((tot_eval_eps_reward/n_eval_eps))
            print("Eval-EPS-AVG: {},".format((tot_eval_eps_reward/n_eval_eps)))


        # In case user wants to stop the training
        if user_input is not None and len(user_input) > 0 and user_input[0] is not None and len(user_input[0]) > 0 and (user_input[0][0]).lower() == 's':
            # Save rewards history; total number of episodes; current ep number
            save_last_run_info(last_run_filename, ep, n_eps, rewards_dict, ddpgAgent, tot_ep_reward_history,
                               tot_eval_ep_avg_reward_history)
            # Save replay memory to disk
            save_last_run_replay_memory(last_run_replay_memory_filename, ddpgAgent)
            # Save agent models checkpoint
            print("Saving models...")
            ddpgAgent.save_models_checkpoints_to_disk()

            plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history)

            break

        total_reward_per_ep = 0
        observation = env.reset()
        done = False
        ep_steps = 0

        while not done:
            action = ddpgAgent.get_action(observation, rewards_dict)
            next_observation, reward, done, info = env.step(action)
            #rewards_dict["now"] = reward
            ddpgAgent.save_state([observation, action, reward, next_observation, done])
            ddpgAgent.train(done)
            observation = next_observation
            total_reward_per_ep += reward
            #steps += 1
            ep_steps += 1
            #To speed up training max number of steps are 400
            if ep_steps == max_ep_steps:
                done = True
            if done:
                #print("Ep-N: {}, Ep-Steps: {}, Tot-Ep-R: {:.5f}, Avg-R-100-EP: {:.5f},Avg-R-10-EP: {:.5f}, Avg-R-{}-EP: {:.5f}, Max-R: {:.3f}, Var-R: {:.3f}, STD-R: {:.3f}, EPS:{:.5f}, EPS_VDBE_R_MT_LT:{:.5f}, EPS_VDBE_R_MT_LT_MTR:{:.2f}, EPS_VDBE_R_MT_LT_LTR:{:.2f}".format(ep, ep_steps, total_reward_per_ep,  rewards_dict["avg_100_ep"], rewards_dict["avg_10_ep"], ep, rewards_dict["avg_tot_ep"], rewards_dict["max"], rewards_dict["variance"], rewards_dict["std"], ddpgAgent.eps, ddpgAgent.eps_vdbe_mt_lt, rewards_dict["mt"], rewards_dict["lt"]))
                print("Ep-N: {}, Ep-Steps: {}, Tot-Ep-R: {:.5f}, Avg-R-100-EP: {:.5f},Avg-R-10-EP: {:.5f}, Avg-R-{}-EP: {:.5f}, EPS:{:.5f}".format(ep, ep_steps, total_reward_per_ep,  rewards_dict["avg_100_ep"], rewards_dict["avg_10_ep"], ep, rewards_dict["avg_tot_ep"], ddpgAgent.eps))
                print("Input [S or s] to STOP - [P or p] to PAUSE")
            if ep % ep_render_step == 0:
                env.render()

        # if ep == 0 and init_max_reward:
        #     rewards_dict["max"] = total_reward_per_ep
        #     init_max_reward = False
        # else:
        #     if total_reward_per_ep > rewards_dict["max"]:
        #         rewards_dict["max"] = total_reward_per_ep
        tot_ep_reward_history.append(total_reward_per_ep)

        rewards_dict["avg_10_ep"] = np.mean(tot_ep_reward_history[-10:])
        rewards_dict["avg_100_ep"] = np.mean(tot_ep_reward_history[-100:])
        rewards_dict["avg_tot_ep"] = np.mean(tot_ep_reward_history[-ep:])
        # if rewards_dict["avg_tot_ep_max"] == -10000000000 or rewards_dict["avg_tot_ep_max"] < rewards_dict["avg_tot_ep"]:
        #     rewards_dict["avg_tot_ep_max"] = rewards_dict["avg_tot_ep"]
        #
        # if rewards_dict["avg_tot_ep_min"] == 10000000000 or rewards_dict["avg_tot_ep_min"] > rewards_dict["avg_tot_ep"]:
        #     rewards_dict["avg_tot_ep_min"] = rewards_dict["avg_tot_ep"]

        # if len(tot_ep_reward_history) >= 2:
        #     rewards_dict["variance"] = statistics.variance(tot_ep_reward_history)
        #     rewards_dict["std"] = math.sqrt(rewards_dict["variance"])
    plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history)

# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     env_name = "CartPole-v1"
#     env = gym.make(env_name)
#     print("Observation space:", env.observation_space)
#     print("Action space:", env.action_space)
#     dqagent = DeepQAgent(env, device, "dqn-main-cartpole", "dqn-target-cartpole")
#     total_reward = 0
#     steps = 0
#     for ep in range(10):
#         total_reward_per_ep = 0
#         observation = env.reset()
#         done = False
#         while not done:
#             action = dqagent.get_action(observation)
#             next_observation, reward, done, info = env.step(action)
#             #dqagent.save_state([observation, action, reward, next_observation, done])
#             #dqagent.train(steps, done)
#             observation = next_observation
#             total_reward += reward
#             total_reward_per_ep += reward
#             steps += 1
#             #print("State:", observation, "Action:", action)
#             if done:
#                 print("Episode: {}, Episode reward: {}, Total reward: {}, Epsilon: {}".format(ep, total_reward_per_ep, total_reward, dqagent.eps))
#                 #if ep % 1000 == 0:
#                     #print("Loss per 1000 episodes: {}".format(dqagent.running_loss/1000))
#             env.render()
#             time.sleep(0.001)
#             #clear_output(wait=True)
