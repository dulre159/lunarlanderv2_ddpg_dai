# For every configuration get eval data
# Average eval data for each configuration - compute std
# Plot the averaged data over the plot with specific color and symbol
# Do the same for all others configurations
import os
import ast
import sys

import numpy as np
import matplotlib.pyplot as plt

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
env_name = "LunarLanderContinuous-v2"
confs = [
    {
        "name": "Gaussian Noise",
        "strategy_name":"just_gnoise",
        "plot_color":"green",
        "plot_symbol":"*",
        "params_folder_name":"std0p1",
        "avg_ep_rewards_history":[],
        "avg_eval_ep_rewards_history":[],
        "avg_ep_rewards_history_stds":[],
        "avg_ep_rewards_history_means":[],
        "avg_eval_ep_rewards_history_stds":[],
        "avg_eval_ep_rewards_history_means":[]
    },
    {
        "name": "Epsilon Greedy",
        "strategy_name": "eps_greedy",
        "plot_color": "magenta",
        "plot_symbol": "1",
        "params_folder_name": "eps-init0p1",
        "avg_ep_rewards_history": [],
        "avg_eval_ep_rewards_history": [],
        "avg_ep_rewards_history_stds": [],
        "avg_ep_rewards_history_means": [],
        "avg_eval_ep_rewards_history_stds": [],
        "avg_eval_ep_rewards_history_means": []
    },
    {
        "name": "No Noise",
        "strategy_name":"no-noise",
        "plot_color":"red",
        "plot_symbol":"D",
        "params_folder_name":"noparams",
        "avg_ep_rewards_history":[],
        "avg_eval_ep_rewards_history":[],
        "avg_ep_rewards_history_stds":[],
        "avg_ep_rewards_history_means":[],
        "avg_eval_ep_rewards_history_stds":[],
        "avg_eval_ep_rewards_history_means":[]
    },
    {
        "name": "Random",
        "strategy_name":"random",
        "plot_color": "blue",
        "plot_symbol": "X",
        "params_folder_name":"noparams",
        "avg_ep_rewards_history":[],
        "avg_eval_ep_rewards_history":[],
        "avg_ep_rewards_history_stds":[],
        "avg_ep_rewards_history_means":[],
        "avg_eval_ep_rewards_history_stds":[],
        "avg_eval_ep_rewards_history_means":[]
    },
    {
        "name": "OUNoise",
        "strategy_name":"ounoise",
        "plot_color": "orange",
        "plot_symbol": "o",
        "params_folder_name":"mu0p0_theta0p15_sigma0p2_dt0p01",
        "avg_ep_rewards_history":[],
        "avg_eval_ep_rewards_history":[],
        "avg_ep_rewards_history_stds":[],
        "avg_ep_rewards_history_means":[],
        "avg_eval_ep_rewards_history_stds":[],
        "avg_eval_ep_rewards_history_means":[]
    },
]
def print_final_plots():
    # Read last run files
    for i in range(0, 5):
        for confdir in confs:
            #last_ep, tot_ep, eps, avg_ep_rewards_history, avg_eval_ep_rewards_history = load_last_run_info()
            last_run_info_file_path = confdir["strategy_name"]+"/"+confdir["params_folder_name"]+"/"+"run"+str(i)+"/last_runs/"+env_name+"_"+confdir["strategy_name"]+"_"+confdir["params_folder_name"]+"_last_run.txt"
            data = load_last_run_info(last_run_info_file_path)
            if data is not None:
                confdir["avg_ep_rewards_history"].append(data[3])
                confdir["avg_eval_ep_rewards_history"].append(data[4])

    # Compute avg eval reward over avg eval rewards history
    for confdir in confs:
            mean_arh = np.mean(confdir["avg_ep_rewards_history"], axis=0)
            mean_aerh = np.mean(confdir["avg_eval_ep_rewards_history"], axis=0)
            std_arh = np.mean(confdir["avg_ep_rewards_history"], axis=0)
            std_aerh = np.mean(confdir["avg_eval_ep_rewards_history"], axis=0)
            confdir["avg_ep_rewards_history_means"] = mean_arh
            confdir["avg_ep_rewards_history_stds"] = std_arh
            confdir["avg_eval_ep_rewards_history_means"] = mean_aerh
            confdir["avg_eval_ep_rewards_history_stds"] = std_aerh

    # Plot on one figure
    x_train = [i+1 for i in range(1000)]
    x_eval = [i for i in range(0, 1001, 100)]
    eval_x_err = [50 for i in range(0, 11)]
    final_results_folder_name = "final_result_plots_output"
    plot_filename_eval = final_results_folder_name + "/eval_final_results.png"
    plot_filename_train = final_results_folder_name + "/train_final_results.png"
    figure_eval, axis_eval = plt.subplots(figsize=(12.8,9.6))
    # figure_eval, axis_eval = plt.subplots()
    # figure_train, axis_train = plt.subplots()
    # axis_eval.set_xlabel('Episodes', fontsize=20)
    axis_eval.set_xlabel('Episodes')
    # axis_eval.set_ylabel('Means eval rewards over all configurations', fontsize=20)
    axis_eval.set_ylabel('Means eval rewards over all configurations')
    for confdir in confs:
        # axis_eval.plot(x_eval,
        #                confdir["avg_eval_ep_rewards_history_means"],
        #                color=confdir["plot_color"],
        #                marker=confdir["plot_symbol"],
        #                linestyle='-',
        #                label=confdir["name"])
        # axis_eval.fill_between(x_eval,
        #                        confdir["avg_eval_ep_rewards_history_stds"]*-1,
        #                        confdir["avg_eval_ep_rewards_history_stds"],
        #                        alpha=0.2,
        #                        edgecolor='#1B2ACC',
        #                        facecolor=confdir["plot_color"],
        #                        linewidth=1,
        #                        linestyle='dashdot',
        #                        antialiased=True)
        axis_eval.errorbar(x_eval,
                           confdir["avg_eval_ep_rewards_history_means"],
                           yerr=confdir["avg_eval_ep_rewards_history_stds"],
                           #xerr=eval_x_err,
                           fmt='',
                           capsize=5,
                           color=confdir["plot_color"],
                           ecolor=confdir["plot_color"],
                           marker=confdir["plot_symbol"],
                           linestyle='-',
                           label=confdir["name"])
    # axis_eval.legend(bbox_to_anchor=(1,1), fontsize=20)
    axis_eval.legend(bbox_to_anchor=(1,1))
    figure_eval.tight_layout()
    figure_eval.savefig(plot_filename_eval)
    plt.close(figure_eval)

if __name__ == '__main__':
    print_final_plots()
    sys.exit()
