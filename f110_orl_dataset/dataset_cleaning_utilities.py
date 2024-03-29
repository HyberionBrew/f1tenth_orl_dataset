import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

# argument for the input zarr
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--input_folder', type=str, default='dataset.zarr', help='.zarr dir to work with')



def plot_agents(root, model_names=None):
    if model_names is None:
        model_names = np.unique(root["model_name"])

    # Get starts array
    ends =  root["truncated"][:] | root["done"][:]
    print(sum(ends))
    starts = np.roll(ends, 1)

    # Initialize color map and legend
    colors = cm.rainbow(np.linspace(0, 1,len(model_names)))
    #legend_handles = []
    for idx, model_name in enumerate(model_names):
        # Extract relevant starts for the current model
        where_model = np.where(root["model_name"][:] == model_name)[0]
        where_starts = np.where(starts[:] == True)[0]

        intersect = np.intersect1d(where_model, where_starts)

        # Create legend entry for this model
        legend_line = mlines.Line2D([], [], color=colors[idx], marker='x', linestyle='-', label=model_name)
        #legend_handles.append(legend_line)

        print(f"Model: {model_name}, Number of Starts: {len(intersect)}")

        for i in intersect:
            # Extract and plot poses
            poses_x = root["observations"]["poses_x"][:]
            poses_y = root["observations"]["poses_y"][:]

            plt.plot(poses_x[i], poses_y[i], 'x', color=colors[idx], scalex=3.0)
            plt.plot(poses_x[i:i+250], poses_y[i:i+250], linestyle='--', color=colors[idx])

            # Plot all terminal states in black for the current model
            terminals = np.where(root["done"][i:i+250] == True)[0]
            #print(f"Terminals at indices {i}: {terminals}")
            for j in terminals:
                plt.plot(poses_x[i+j], poses_y[i+j], 'o', color="black", scalex=3.0)
        plt.title("Model Name: " + model_name)
        plt.show()


def plot_agent_trajectory(root, model_name, full=False,show_term_steps=False):
    ends =  root["truncated"][:] | root["done"][:]
    starts = np.roll(ends, 1)

    #legend_handles = []
    # Extract relevant starts for the current model
    where_model = np.where(root["model_name"][:] == model_name)[0]
    where_starts = np.where(starts[:] == True)[0]
    where_ends = np.where(ends[:] == True)[0]
    intersect_starts = np.intersect1d(where_model, where_starts)
    intersect_ends = np.intersect1d(where_model, where_ends)
    poses_x = root["observations"]["poses_x"][:]
    poses_y = root["observations"]["poses_y"][:]
    terminals = root["done"][:]
    print("starts;", intersect_starts)
    xx = np.unique(poses_x[intersect_starts],return_counts=True)
    #print(len(xx[1]))
    #print(xx[1])
    print(f"Agent {model_name}, Number of Starts: {len(intersect_starts)}.")
    for start, end in zip(intersect_starts, intersect_ends):
        # Extract and plot poses

        #print(start)
        #plt.plot(poses_x[start:end], poses_y[start:end], linestyle='--', color="black")
        if show_term_steps:
            plt.plot(poses_x[end-25:end], poses_y[end-25:end], 'o', color="blue")
        plt.plot(poses_x[start], poses_y[start], 'x', color="black", scalex=3.0)
        plt.plot(poses_x[end], poses_y[end], 'o', color="grey", scalex=3.0)
        if terminals[end]:
            plt.plot(poses_x[end], poses_y[end], 'o', color="red", scalex=3.0)
        
       
        if not(full):
            plt.title("Model Name: " + model_name)
            plt.show()
    if full:
        plt.title("Model Name: " + model_name)
        plt.show()
    return intersect_starts
def plot_multiple_trajectories(root, model_names, full=True, show_term_steps=False):
    if model_names is None:
        model_names = np.unique(root["model_name"])
    starts = []
    for name in model_names:
        print(name)
        starts.append(plot_agent_trajectory(root, name, full, show_term_steps))
    return starts



def plot_trajectory(dataset, trajectory_start_index, axs, title = None):
    trajectory_length = 251
    
    terminals = dataset["done"][trajectory_start_index:trajectory_start_index + trajectory_length]
    truncated = dataset["truncated"][trajectory_start_index:trajectory_start_index + trajectory_length]
    end = terminals | truncated
    # find first end 
    first_end = np.where(end)[0][0]
    trajectory_length = first_end + 1
    trajectory_data = {key: dataset["observations"][key][trajectory_start_index:trajectory_start_index + trajectory_length] for key in dataset["observations"].keys() if key != "lidar_occupancy"}
    trajectory_data["terminals"] = dataset["done"][trajectory_start_index:trajectory_start_index + trajectory_length]
    trajectory_data["truncated"] = dataset["truncated"][trajectory_start_index:trajectory_start_index + trajectory_length]
    # Extract trajectory data
    

    trajectory_data["raw_actions"] = dataset["raw_actions"][trajectory_start_index:trajectory_start_index + trajectory_length]
    # add action and terminals
    trajectory_data["actions"] = dataset["actions"][trajectory_start_index:trajectory_start_index + trajectory_length]
    trajectory_data["terminals"] = dataset["done"][trajectory_start_index:trajectory_start_index + trajectory_length]
    trajectory_data["truncated"] = dataset["truncated"][trajectory_start_index:trajectory_start_index + trajectory_length]
    # Plot the trajectory data
   

    for i, (key, data) in enumerate(trajectory_data.items()):
        axs[i].plot(data)
        axs[i].set_title(key)
        axs[i].set_xlabel('Time step')
        axs[i].set_ylabel(key)


   
def plot_multiple_trajectories_data(dataset, start_indices, title = None):
    values = len(dataset["observations"]) - 1 + 4
    fig, axs = plt.subplots(values, 1, figsize=(10, values * 3))
    # fig.suptitle(f'Trajectory {trajectory_start_index}') if title is None else fig.suptitle(title)
    for start_index in start_indices:
        plot_trajectory(dataset, start_index, axs, title)
    plt.tight_layout()

    plt.show()


def get_erroneous_terminals(dataset, trajectory_length=250):
    ends = dataset["truncated"][:]
    agent_starts = np.where(np.roll(ends, 1)==True)[0]
    wrong_terminals = []
    for trajectory_start_index in agent_starts:
        specific_dones = dataset["done"][trajectory_start_index:trajectory_start_index + trajectory_length]
        num_dones = np.count_nonzero(specific_dones)
        
        if num_dones > 0 and num_dones < 3:
            #print(num_dones)
            plot_trajectory(dataset, trajectory_start_index)
            print(f"Number of dones: {num_dones}")
            wrong_terminals.append(trajectory_start_index)
            

    return wrong_terminals

def set_erroneous_terminals(dataset, trajectory_length=250):
    ends = dataset["truncated"][:]
    agent_starts = np.where(np.roll(ends, 1) == True)[0]

    for trajectory_start_index in agent_starts:
        # Extract the specific 'done' array for the trajectory
        specific_dones = dataset["done"][trajectory_start_index:trajectory_start_index + trajectory_length]
        # check if there are any dones, if not skip
        if np.count_nonzero(specific_dones) == 0:
            continue
        first_done_index = np.argmax(specific_dones)
        new_dones = np.zeros_like(specific_dones)

        new_dones[first_done_index:] = True
        if (new_dones != specific_dones).any():
            dataset["done"][trajectory_start_index + first_done_index:trajectory_start_index + trajectory_length +1] = True
            print("initally", np.where(specific_dones == True))
            print(f"Corrected terminals in trajectory starting at index {trajectory_start_index}")
            print(f"{trajectory_start_index + first_done_index}:{trajectory_start_index + trajectory_length}")
        # print the indices of previous dones


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = zarr.open(args.input_folder, mode='r')
    uniques = np.unique(dataset["model_name"])
    print(uniques)
    print(np.where(dataset["collision"]))
    print(np.where(dataset["done"]))
    for i in np.where(dataset["collision"])[0]:
        # print where not the same
        assert (dataset["collision"][i] == dataset["done"][i]), f"Collision and done not the same {i}"
    # assert((dataset["collision"] == dataset["done"]))
    #agent_names =  ["StochasticContinousFTGAgent_0.55_3_0.5_0.03_0.1_5.0_0.3_0.5"]
    #agent_names = ["StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5"]#,"StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5" ] # StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5
    # agent_names = ["pure_pursuit2_0.8_1.0_raceline6_c_0.3_0.5"]
    #plot_trajectory(dataset, 67_579, 5)
    #plot_agents(dataset, agent_names)
    #print(sum(dataset["truncated"])+ sum(dataset["done"]))
    ends =  dataset["truncated"][:] | dataset["done"][:]
    starts = np.roll(ends, 1)
    print(dataset["observations"].keys())
    print(dataset["observations"]["poses_x"].shape)
    start_idx = np.where(starts)[0]
    model_names = np.unique(dataset["model_name"])
    eval_agents =  ["pure_pursuit2_0.6_1.0_raceline2_0.3_0.5",
                                    "pure_pursuit2_0.7_0.9_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.8_0.95_raceline3_0.3_0.5",
                                    # "pure_pursuit2_0.8_1.25_raceline8_0.3_0.5", # only has 49 trajectories :/
                                    "pure_pursuit2_0.44_0.85_raceline1_0.3_0.5",
                                    "pure_pursuit2_0.52_0.9_raceline4_0.3_0.5",
                                    "pure_pursuit2_0.65_1.2_centerline_0.3_0.5",
                                    "pure_pursuit2_0.68_1.1_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.73_0.95_centerline_0.3_0.5",
                                    "pure_pursuit2_0.73_0.95_raceline4_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.8_2_0.7_0.03_0.1_5.0_0.3_0.5",
                                  #   "StochasticContinousFTGAgent_1.0_0_0.2_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_1.0_1_0.2_0.03_0.1_5.0_0.3_0.5",
                                    "pure_pursuit2_0.4_0.3_raceline4_0.3_0.5",
                                    "pure_pursuit2_0.44_0.3_raceline1_0.3_0.5"]
    for name in eval_agents:
        intersect = np.intersect1d(np.where(dataset["model_name"][:] == name)[0], np.where(starts == True)[0])
        print(name)
        plt.scatter(dataset["observations"]["poses_x"][start_idx], dataset["observations"]["poses_y"][start_idx], color="blue")
        plt.scatter(dataset["observations"]["poses_x"][intersect], dataset["observations"]["poses_y"][intersect], color="red")
        # annotate the intersects with the index
        for i in intersect:
            plt.annotate(str(i), (dataset["observations"]["poses_x"][i], dataset["observations"]["poses_y"][i]))
            
        plt.show()
    
    agent_names = None # ["StochasticContinousFTGAgent_0.45_3_0.5_0.03_0.1_5.0_0.3_0.5"]#None#["pure_pursuit2_0.4_0.3_raceline4_0.3_0.5"]
    # plot all starts
    starts = plot_multiple_trajectories(dataset, agent_names, full=True, show_term_steps=False)
    print(starts)
    print(len(starts))
    for start in starts:
        for tra in start:
            print(tra)
            plot_multiple_trajectories_data(dataset, [tra])
    #plot_agent_trajectory(dataset, agent_names[0], full=True, show_term_steps=False)
    #
    """
    print(dataset["collision"][67017+249])
    print(dataset["collision"][67017+248])
    print(dataset["collision"][67017+250])

    print(dataset["truncated"][67017+249])
    print(dataset["truncated"][67017+248])
    print(dataset["truncated"][67017+250])
    """
    # print(set_erroneous_terminals(dataset))
    # dataset["collision"] = dataset["done"]
    #dataset["collision"][67017+248] = False
    #dataset["collision"][67017+249] = False
    #dataset["done"] = dataset["collision"]
    """
    print(np.where(dataset["truncated"]))
    dists = np.where(dataset["truncated"])[0]- np.roll(np.where(dataset["truncated"])[0], 1)#[260:271]
    print(np.diff(np.where(dataset["truncated"])[0]))
    print("llö")
    print(dists)
    print(np.where(dists!=251))
    #print(np.where(np.where(dataset["truncated"])- np.roll(np.where(dataset["truncated"]), 1)[0]!=251))
    #print(uniques)
    print(dataset["model_name"][269*251])
    print(dataset["model_name"][267*251])
    #plot_trajectory(dataset, 267*251)
    #plot_trajectory(dataset, 268*251)
    #plot_trajectory(dataset, 269*251)
    plt.plot(dataset["truncated"][269*251 - 1:271*251])
    plt.show()
    print(uniques[0])
    print(type(uniques[0]))
    print(np.where(uniques=="pure_pursuit_0.6_1.0_raceline_og_3")[0])
    print(uniques[18])
    print(uniques[18] == "pure_pursuit_0.6_1.0_raceline_og_3")
    #print(np.count_nonzero(dataset["model_name"] == 'StochasticContinousFTGAgent_0.65_0_0.2_0.15_0.15_5.0_0.1'))
    print(dataset["model_name"])
    indices = np.where(np.array(dataset["model_name"]) == uniques[18])[0]
    #print(dataset["model_name"][:10])
    print(indices)
    print(len(indices))
    plt.scatter(dataset["observations"]["poses_x"][indices], dataset["observations"]["poses_y"][indices])
    plt.show()
    #plot_trajectory(dataset, agent_names[0], 5)
    # plot_trajectory(dataset, agent_names[0], 6)
    
    #indices_of_wrong_terminals = get_erroneous_terminals(dataset)
    #print(indices_of_wrong_terminals)

    #plot_trajectory(dataset, agent_names[0], 7)
    # find indices where there is only 0-3 timeouts
    """