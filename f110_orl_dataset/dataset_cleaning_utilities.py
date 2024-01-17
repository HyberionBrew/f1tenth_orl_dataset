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
    ends =  root["truncated"][:]
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
            print(f"Terminals at indices {i}: {terminals}")
            for j in terminals:
                plt.plot(poses_x[i+j], poses_y[i+j], 'o', color="black", scalex=3.0)
        plt.title("Model Name: " + model_name)
        plt.show()

def plot_trajectory(dataset, trajectory_start_index, title = None):
    trajectory_length = 251

    # Extract trajectory data
    trajectory_data = {key: dataset["observations"][key][trajectory_start_index:trajectory_start_index + trajectory_length] for key in dataset["observations"].keys() if key != "lidar_occupancy"}

    trajectory_data["raw_actions"] = dataset["raw_actions"][trajectory_start_index:trajectory_start_index + trajectory_length]
    # add action and terminals
    trajectory_data["actions"] = dataset["actions"][trajectory_start_index:trajectory_start_index + trajectory_length]
    trajectory_data["terminals"] = dataset["done"][trajectory_start_index:trajectory_start_index + trajectory_length]
    trajectory_data["truncated"] = dataset["truncated"][trajectory_start_index:trajectory_start_index + trajectory_length]
    # Plot the trajectory data
    fig, axs = plt.subplots(len(trajectory_data), 1, figsize=(10, len(trajectory_data) * 3))
    fig.suptitle(f'Trajectory {trajectory_start_index}') if title is None else fig.suptitle(title)

    for i, (key, data) in enumerate(trajectory_data.items()):
        axs[i].plot(data)
        axs[i].set_title(key)
        axs[i].set_xlabel('Time step')
        axs[i].set_ylabel(key)

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
    print(np.where(dataset["collision"]))
    print(np.where(dataset["done"]))
    for i in np.where(dataset["collision"])[0]:
        # print where not the same
        assert (dataset["collision"][i] == dataset["done"][i]), f"Collision and done not the same {i}"
    # assert((dataset["collision"] == dataset["done"]))
    agent_names = None
    agent_names = ["pure_pursuit_0.9_1.4_raceline_og_3"]
    #plot_trajectory(dataset, 67_579, 5)
    plot_agents(dataset, agent_names)
    uniques = np.unique(dataset["model_name"])
    print(dataset["collision"][67017+249])
    print(dataset["collision"][67017+248])
    print(dataset["collision"][67017+250])

    print(dataset["truncated"][67017+249])
    print(dataset["truncated"][67017+248])
    print(dataset["truncated"][67017+250])
    print(set_erroneous_terminals(dataset))
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