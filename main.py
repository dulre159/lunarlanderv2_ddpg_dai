import os.path
import signal
import sys
import gym
import numpy as np
from ddpg_agent import DDPGAgent
import time
from main_utils import get_user_input, exit_gracefully, load_last_run_replay_memory
from main_utils import load_last_run_info, plot_running_mean_of_rewards_history, plot_eval_mean_rewards_history
from main_utils import save_everything, do_evaluation, search_if_dir_has_file_including_subdirs

original_sigint = signal.getsignal(signal.SIGINT)
user_input = [None]

if __name__ == '__main__':

    # signal.signal(signal.SIGINT, exit_gracefully, user_input, original_sigint)
    #
    # input_thread = threading.Thread(target=get_user_input, args=(user_input,))
    # input_thread.daemon = True
    # input_thread.start()

    # Choose exploration vs exploitation strategy
    # exp_exp_strategy_name = "ounoise"
    # exp_exp_strategy_name = "just_gnoise"
    # exp_exp_strategy_name = "gnoise_eps-decay"
    # exp_exp_strategy_name = "eps_greedy"
    # exp_exp_strategy_name = "eps_greedy_eps-decay"
    # exp_exp_strategy_name = "random"
    exp_exp_strategy_name = "adaptive-parameter-noise"
    # exp_exp_strategy_name = "no-noise"

    exp_exp_strategy_filename = exp_exp_strategy_name

    # Make strategy folder
    if not os.path.isdir(exp_exp_strategy_name):
        os.mkdir(exp_exp_strategy_name)

    load_run = False
    # Can be last_run or runX
    run_to_load_name="last_run"
    load_models_from_disk = False
    load_last_run_from_disk = False
    load_last_run_replay_memory_from_disk = False

    if load_run is True and load_models_from_disk is False and load_last_run_from_disk is False or load_last_run_replay_memory_from_disk is False:
        sys.exit("Aborting... To load a run you must load at least its model or replay memory or last_run_info_file...")

    env_name = "LunarLanderContinuous-v2"
    # env_name = "BipedalWalker-v3"
    env = gym.make(env_name)

    rewards_dict ={"avg_10_ep": 0, "avg_50_ep": 0, "avg_100_ep": 0, "avg_tot_ep":0, "now": 0, "max": 0, "variance" : 0, "std" : 0}

    eps_init=0.1 if "eps_greedy" in exp_exp_strategy_name else 1.0
    eps_min=0.001
    eps_dec= 0.00009899 if "eps_greedy" in exp_exp_strategy_name else 0.000999

    gnoisestd=0.1

    ounMU=0.0
    ounTheta=0.15
    ounSigma=0.2
    ounDT=1e-2

    apnInitialStddev = 0.05
    apnDesiredActionStddev = 0.7
    # apnAdaptationCoefficient = 1.01
    apnAdaptationCoefficient = 0.99

    strategy_params=""
    # Change filename depending on the selected strategy
    if exp_exp_strategy_name == "just_gnoise":
        strategy_params += "std" + str(gnoisestd).replace(".", "p")
    elif exp_exp_strategy_name == "gnoise_eps-decay":
        strategy_params +=  \
        "std" + str(gnoisestd).replace(".", "p") + "_" + \
        "eps-init" + str(eps_init).replace(".", "p") + "_" + \
        "eps-dec" + str(eps_dec).replace(".", "p") + "_" + \
        "eps-min" + str(eps_min).replace(".", "p")
    elif exp_exp_strategy_name == "eps_greedy":
        strategy_params +=  \
        "eps-init" + str(eps_init).replace(".", "p")
    elif exp_exp_strategy_name == "eps_greedy_eps-decay":
        strategy_params +=  \
        "eps-init" + str(eps_init).replace(".", "p") + "_" \
        "eps-dec" + str(eps_dec).replace(".", "p") + "_" + \
        "eps-min" + str(eps_min).replace(".", "p")
    elif exp_exp_strategy_name == "ounoise":
        strategy_params +=  \
        "mu" + str(ounMU).replace(".", "p") + "_" \
        "theta" + str(ounTheta).replace(".", "p") + "_" + \
        "sigma" + str(ounSigma).replace(".", "p") + "_" + \
        "dt" + str(ounDT).replace(".", "p")
    elif exp_exp_strategy_name == "adaptive-parameter-noise":
        strategy_params +=  \
        "apnInitialStddev" + str(apnInitialStddev).replace(".", "p") + "_" \
        "apnDesiredActionStddev" + str(apnDesiredActionStddev).replace(".", "p") + "_" + \
        "apnAdaptationCoefficient" + str(apnAdaptationCoefficient).replace(".", "p")
    else:
        strategy_params = "noparams"
    exp_exp_strategy_filename += "_" + strategy_params

    strategy_params_folder_path = exp_exp_strategy_name+"/"+strategy_params

    run_name = ""

    # Create folder for strategy params
    if not os.path.isdir(strategy_params_folder_path):
        os.mkdir(strategy_params_folder_path)

    # Get last run folder number
    runs_folders = []
    for searched_dir_path, dirnames, filenames in os.walk(strategy_params_folder_path):
        runs_folders = dirnames
        break

    if runs_folders is None or len(runs_folders)==0:
        if load_run:
            sys.exit("Aborting... \nThere is no run to load data from for strategy: "+"["+exp_exp_strategy_name+"] with params: "+"["+strategy_params+"]....")
        else:
            # Create run0 folder for strategy params
            if not os.path.isdir(strategy_params_folder_path+"/run0"):
                os.mkdir(strategy_params_folder_path+"/run0")
                run_name = "run0"
    elif load_run and run_to_load_name != "last_run" and run_to_load_name != "":
        if run_to_load_name in runs_folders:
            run_name = run_to_load_name
        else:
            sys.exit("Aborting...\nCould not load run \""+run_to_load_name+"\"...")
    else:
        runs_folders.sort()
        run_name = str(runs_folders[-1])

    # If last run folder is totally empty use it otherwise create a new run folder for the new run only if load_run is false
    a = search_if_dir_has_file_including_subdirs(strategy_params_folder_path+"/"+run_name)
    if a is True and load_run is False:
        str_run_num = run_name.replace("run", "")
        int_run_num = int(str_run_num)
        int_run_num+=1
        run_name = "run"+str(int_run_num)
        # Create new run folder for strategy params
        if not os.path.isdir(strategy_params_folder_path + "/"+run_name):
            os.mkdir(strategy_params_folder_path + "/"+ run_name)


    strategy_params_plots_folder_path = strategy_params_folder_path+"/"+run_name+'/plots'
    strategy_params_last_runs_folder_path = strategy_params_folder_path+"/"+run_name+'/last_runs'
    strategy_params_checkpoints_folder_path = strategy_params_folder_path+"/"+run_name+'/checkpoints'

    # Create needed directories if they do not exist
    if not os.path.isdir(strategy_params_plots_folder_path):
        os.mkdir(strategy_params_plots_folder_path)
    if not os.path.isdir(strategy_params_last_runs_folder_path):
        os.mkdir(strategy_params_last_runs_folder_path)
    if not os.path.isdir(strategy_params_checkpoints_folder_path):
        os.mkdir(strategy_params_checkpoints_folder_path)

    plot_filename = strategy_params_plots_folder_path+ '/' + env_name + "_" + exp_exp_strategy_filename + "_"
    eval_plot_filename = strategy_params_plots_folder_path+'/eval_plot_' + env_name + "_" + exp_exp_strategy_filename + "_"
    last_run_filename = strategy_params_last_runs_folder_path+'/' + env_name+"_"+exp_exp_strategy_filename+"_last_run.txt"
    last_run_replay_memory_filename = strategy_params_last_runs_folder_path+'/' + env_name+"_"+exp_exp_strategy_filename+"_last_run_replay_memory.bin"
    actor_chkp_file_path = strategy_params_checkpoints_folder_path + "/ddpg-actor_"+env_name+"_"+exp_exp_strategy_filename
    actor_target_chkp_file_path = strategy_params_checkpoints_folder_path + "/ddpg-actor-target_"+env_name+"_"+exp_exp_strategy_filename
    critic_chkp_file_path = strategy_params_checkpoints_folder_path + "/ddpg-critic_" + env_name + "_" + exp_exp_strategy_filename
    critic_target_chkp_file_path = strategy_params_checkpoints_folder_path + "/ddpg-critic-target_" + env_name + "_" + exp_exp_strategy_filename

    last_run_information = None
    last_run_replay_memory = None

    if load_last_run_from_disk:
        # Result of the below func will be None or a tuple like (ep, n_eps, epsilon, tot_ep_reward_history, tot_eval_ep_avg_reward_history)
        last_run_information = load_last_run_info(last_run_filename)

        if last_run_information is not None:
            rewards_dict["avg_10_ep"] = np.mean(last_run_information[3][-10:])
            rewards_dict["avg_100_ep"] = np.mean(last_run_information[3][-100:])
            rewards_dict["avg_tot_ep"] = np.mean(last_run_information[3][-last_run_information[0]:])
            eps_init = last_run_information[2]

    if load_last_run_replay_memory_from_disk:
        last_run_replay_memory = load_last_run_replay_memory(last_run_replay_memory_filename)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    ddpgAgent = DDPGAgent(
        env,
        actor_chkp_file_path,
        actor_target_chkp_file_path,
        critic_chkp_file_path,
        critic_target_chkp_file_path,
        last_run_replay_memory,
        exp_exp_strategy_name,
        load_models_from_disk,
        gnoisestd=gnoisestd,
        eps_init=eps_init,
        eps_min=eps_min,
        eps_dec=eps_dec,
        ounSigma=ounSigma,
        ounTheta=ounTheta,
        ounMU=ounMU,
        ounDT=ounDT,
        apnInitialStddev=apnInitialStddev,
        apnDesiredActionStddev=apnDesiredActionStddev,
        apnAdaptationCoefficient=apnAdaptationCoefficient
        )

    steps = 0
    n_eps = 1000 if not load_last_run_from_disk or last_run_information is None else last_run_information[1]
    ep = 0 if not load_last_run_from_disk or last_run_information is None else last_run_information[0]
    tot_ep_reward_history = [] if not load_last_run_from_disk or last_run_information is None else last_run_information[3]
    tot_eval_ep_avg_reward_history = [] if not load_last_run_from_disk or last_run_information is None else last_run_information[4]
    ep_reward_best = env.reward_range[0]
    save_ep = 100
    eval_ep = 100
    n_eval_eps = 50
    # To speed up training max number of steps per episode are 300
    max_ep_steps = 300
    max_eval_ep_steps = 300
    ep_render_step = 100

    start_time = time.time()
    print("TRAINING STARTING TIME: "+time.strftime("%d/%m/%Y-%H:%M:%S"))
    print("STRATEGY: "+exp_exp_strategy_name)
    print("EPS-INIT: "+str(eps_init))
    print("EPS-DEC: "+str(eps_dec))
    print("EPS-MIN: "+str(eps_min))
    print("GAUSSIAN_NOISE_STD: "+str(gnoisestd))

    for ep in range(ep, n_eps):

        # Save step
        if ep % save_ep == 0:
            save_everything(ddpgAgent, last_run_replay_memory_filename, last_run_filename, plot_filename, ep, n_eps,
                                rewards_dict, tot_ep_reward_history, tot_eval_ep_avg_reward_history)
            # plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history)
            # plot_eval_mean_rewards_history(eval_plot_filename, ep, tot_eval_ep_avg_reward_history, eval_ep)

        # Evaluation step
        if ep % eval_ep == 0:
            do_evaluation(ddpgAgent, n_eval_eps, env, rewards_dict, max_eval_ep_steps, tot_eval_ep_avg_reward_history)
            # plot_eval_mean_rewards_history(eval_plot_filename, ep, tot_eval_ep_avg_reward_history, eval_ep)

        # In case user wants to stop the training
        # if user_input is not None and len(user_input) > 0 and user_input[0] is not None and len(user_input[0]) > 0 and (user_input[0][0]).lower() == 's':
        #     save_everything(ddpgAgent, last_run_replay_memory_filename, last_run_filename, plot_filename, ep, n_eps,
        #                     rewards_dict, tot_ep_reward_history, tot_eval_ep_avg_reward_history)
        #     plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history)
        #     break

        total_reward_per_ep = 0
        observation = env.reset()
        done = False
        ep_steps = 0

        # Reset OUNoise every episode. Reason unknown...
        ddpgAgent.ounoise.reset()

        while not done:
            action = ddpgAgent.get_action(observation, rewards_dict)
            next_observation, reward, done, info = env.step(action)
            # rewards_dict["now"] = reward
            if ep_steps == max_ep_steps:
                done = True
            ddpgAgent.save_state([observation, action, reward, next_observation, done])
            ddpgAgent.train(steps, done)
            observation = next_observation
            total_reward_per_ep += reward
            steps += 1
            ep_steps += 1
            if done:
                print("Ep-N: {}, Ep-Steps: {}, Tot-Ep-R: {:.5f}, Avg-R-100-EP: {:.5f},Avg-R-10-EP: {:.5f}, Avg-R-{}-EP: {:.5f}, Max-R:{:.5f}, EPS:{:.5f}, APN-Scalar:{:.5f}".format(ep, ep_steps, total_reward_per_ep,  rewards_dict["avg_100_ep"], rewards_dict["avg_10_ep"], ep, rewards_dict["avg_tot_ep"],rewards_dict["max"], ddpgAgent.eps, ddpgAgent.apnNoise.get_current_stddev()))
                print("Input [S or s] to STOP - [P or p] to PAUSE")
            # if ep % ep_render_step == 0:
                # env.render()

        tot_ep_reward_history.append(total_reward_per_ep)

        rewards_dict["avg_10_ep"] = np.mean(tot_ep_reward_history[-10:])
        rewards_dict["avg_100_ep"] = np.mean(tot_ep_reward_history[-100:])
        rewards_dict["avg_tot_ep"] = np.mean(tot_ep_reward_history[-ep:])
        rewards_dict["max"] = np.max(tot_ep_reward_history[-ep:])

        # if len(tot_ep_reward_history) >= 2:
        #     rewards_dict["variance"] = statistics.variance(tot_ep_reward_history)
        #     rewards_dict["std"] = math.sqrt(rewards_dict["variance"])

    do_evaluation(ddpgAgent, n_eval_eps, env, rewards_dict, max_eval_ep_steps, tot_eval_ep_avg_reward_history)
    save_everything(ddpgAgent, last_run_replay_memory_filename, last_run_filename, plot_filename, ep, n_eps,
                    rewards_dict, tot_ep_reward_history, tot_eval_ep_avg_reward_history)
    plot_running_mean_of_rewards_history(plot_filename, ep, tot_ep_reward_history)
    plot_eval_mean_rewards_history(eval_plot_filename, ep, tot_eval_ep_avg_reward_history, eval_ep)

    elapsed_time = time.time() - start_time
    print("TRAINING END TIME: " + time.strftime("%d/%m/%Y-%H:%M:%S"))
    print("TRAINING TOOK: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


# Old code for testing DQNAgent
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
