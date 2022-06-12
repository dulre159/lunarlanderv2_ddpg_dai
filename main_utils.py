import sys
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
import ast
import signal
import os


def get_user_input(user_input_ref):
    user_input_ref[0] = input("Input [S or s] to STOP - [P or p] to PAUSE\n")

def exit_gracefully(signum, frame, user_input, original_sigint):
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
                            str(ddpgAgent.eps) + ";" + \
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
    if not os.path.exists(last_run_filename):
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
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pf = plot_filename + timestr + '.png'
    print("Plotting running mean of rewards history on file: "+str(pf)+"...")
    x = [i + 1 for i in range(ep)]
    plot_learning_curve(x, tot_ep_reward_history, pf)
def plot_eval_mean_rewards_history(eval_plot_filename, ep, tot_eval_ep_avg_reward_history, eval_ep):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    pf = eval_plot_filename + timestr + '.png'
    print("Plotting evaluation mean rewards history on file: "+str(pf)+"...")
    plot_eval_curve(eval_ep, ep, tot_eval_ep_avg_reward_history, pf)

def save_everything(ddpgAgent, last_run_replay_memory_filename, last_run_filename, ep, n_eps, rewards_dict, tot_ep_reward_history, tot_eval_ep_avg_reward_history):
  print("--- Starting Saving Phase ---")
  print("Saving models...")
  ddpgAgent.save_models_checkpoints_to_disk()

  save_last_run_replay_memory(last_run_replay_memory_filename, ddpgAgent)
  save_last_run_info(last_run_filename, ep, n_eps, rewards_dict, ddpgAgent, tot_ep_reward_history, tot_eval_ep_avg_reward_history)
  print("--- End Of Saving Phase ---")

def do_evaluation(ddpgAgent, n_eval_eps, env, rewards_dict, max_eval_ep_steps, tot_eval_ep_avg_reward_history):
  print("--- Starting Evaluation Phase ---")
  tot_eval_eps_reward = 0
  for eep in range(0, n_eval_eps):
      eval_obs = env.reset()
      eval_done = False
      total_eval_ep_reward = 0
      eval_ep_steps = 0
      while not eval_done:
          eval_action = ddpgAgent.get_eval_action(eval_obs)
          eval_next_observation, eval_reward, eval_done, eval_info = env.step(eval_action)
          eval_obs = eval_next_observation
          total_eval_ep_reward += eval_reward
          eval_ep_steps +=1

          # To speed up evaluation max number of steps are 400
          if eval_ep_steps == max_eval_ep_steps:
              eval_done = True

          if eval_done:
              print("Eval-EP: {}, TOT-R: {}, STEPS: {}".format(eep, total_eval_ep_reward, eval_ep_steps))
          #env.render()
      tot_eval_eps_reward += total_eval_ep_reward
  print("--- End Of Evaluation Phase ---")
  tot_eval_ep_avg_reward_history.append((tot_eval_eps_reward/n_eval_eps))
  print("Eval-EPS-AVG: {},".format((tot_eval_eps_reward/n_eval_eps)))


def search_if_dir_has_file_including_subdirs(dir_path):
    for x in os.scandir(dir_path):
        if x.is_file(follow_symlinks=False):
            return True
        if x.is_dir(follow_symlinks=False):
           if search_if_dir_has_file_including_subdirs(x.path) is True:
               return True

    return False


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

def plot_eval_curve(eval_ep, tot_ep, tot_eval_ep_avg_reward_history, plot_filename):
    x = []
    for nn in range(0, tot_ep+2):
        if nn % eval_ep == 0:
            x.append(nn)
    if len(tot_eval_ep_avg_reward_history) == len(x) and len(x)+len(tot_eval_ep_avg_reward_history) > 0:
        fig2, ax2 = plt.subplots()
        ax2.plot(x, tot_eval_ep_avg_reward_history,color='orange', marker='x', linestyle='-')
        ax2.set_xlabel('train episodes')
        ax2.set_ylabel('mean reward over 50 eval ep')
        fig2.savefig(plot_filename)
        plt.close(fig2)