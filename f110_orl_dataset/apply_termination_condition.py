import numpy as np
import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

def plot_trajectories_on_map(yaml_path, poses, terminations):
    # Load map metadata
    with open(yaml_path, 'r') as file:
        map_metadata = yaml.safe_load(file)

    # Construct the path for the map image
    map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
    map_image = Image.open(map_image_path)
    map_array = np.array(map_image)

    # Display the map
    plt.imshow(map_image, cmap='gray')

    # Map parameters
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']

    # Number of trajectories
    num_trajectories = len(poses)

    # Generate color map
    colors = cm.rainbow(np.linspace(0, 1, num_trajectories))

    # Plot each trajectory
    print(len(terminations))
    for i in range(num_trajectories):
        # Convert poses to pixel coordinates, invert y-axis
        pixel_poses = poses[i].copy()
        pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
        pixel_poses[:, 1] = map_array.shape[0] - ((pixel_poses[:, 1] - origin[1]) / resolution)
        # print(terminations[i])
        # Plot trajectory
        term_loc = np.where(terminations[i])[0]
        if term_loc.size > 0:
            term_loc = term_loc[0]
        else:
            term_loc = len(terminations[i])
        plt.plot(pixel_poses[:term_loc+1, 0], pixel_poses[:term_loc+1, 1],linestyle='--' ,color=colors[i],label=f'Trajectory {i+1}')
        # plot all terminal states in red, each terminal at i is an array of length pixel_poses
        #for j in np.where(terminations[i])[0]:
        #    plt.plot(pixel_poses[j,0],pixel_poses[j,1],'x',color="red", scalex=3.0)
        plt.plot(pixel_poses[term_loc,0],pixel_poses[term_loc,1],'x',color="red", scalex=3.0)
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('Trajectories on Map')
    #plt.legend()
    plt.show()
use_sim_laser = False
yaml_path = "/home/fabian/msc/f110_dope/ws_release/f1tenth_gym/gym/f110_gym/maps/Infsaal3/Infsaal3_map.yaml"
import f110_gym
import f110_orl_dataset
import gymnasium as gym
F110Env = gym.make("f110-sim-stoch-v2",
        encode_cyclic=True,
        flatten_obs=True,
        timesteps_to_include=(0,250),
        use_delta_actions=True,
        include_timesteps_in_obs = False,
        set_terminals=True,
        delta_factor=1.0,
        reward_config="reward_progress.json",
        include_pose_time_diff=False,
        include_action_pose_time_diff = False,
        include_time_obs = False,
        include_progress=False,
        set_previous_step_terminals=0,
        use_compute_termination=True,
        remove_cons_terminals=True,
        **dict(name="f110-sim-stoch-v2",
            config = dict(map="Infsaal3", num_agents=1,
            params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    )



with open(yaml_path, 'r') as file:
    map_metadata = yaml.safe_load(file)
map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
map_image = Image.open(map_image_path).convert('L')  # Convert to grayscale
map_array = np.array(map_image)

zarr_path = "/home/fabian/.f110_rl_datasets/f110-real-stoch-v2" #"/home/fabian/.f110_rl_datasets/f110-real-stoch-v2"#"/home/fabian/msc/f110_dope/ws_release/sim_dataset.zarr"

import zarr

root = zarr.open(zarr_path, mode='r+')
ends = np.array(root["done"]) | np.array(root["truncated"])


model_names = np.array(root["model_name"])
print(model_names.shape)
terminal_ends = np.where(np.array(root["done"]))[0]
print(len(np.where(np.roll(ends,1) & ~(ends))[0]))
starts = np.where(np.roll(ends,1))[0]
print(np.where(ends==1)[0].shape)
# exit()
ends = np.where(ends)[0]
true_positves = 0
false_positives = 0
missed = 0
uncrashing_trajectories = 0
trajectories_missed = []
trajectory_terminations_missed = []
trajectory_false_positive = []
trajectory_terminations_false_positive = []
trajectories_true_positives = []
trajectory_terminations_true_positives = []
names_missed = []
new_terminals = np.zeros(len(root["done"]),dtype=bool)
i = 0
for start, end in tqdm(zip(starts, ends), total=len(starts)):
    i += 1
    target_speed = np.array(root["raw_actions"])
    #print(target_speed.shape)
    #print(start,end)
    xy = np.concatenate((np.array(root["observations"]["poses_x"][start:end+1]).reshape(-1,1), 
                        np.array(root["observations"]["poses_y"][start:end+1]).reshape(-1,1))
                        ,axis=1)
    
    #print(xy.shape)
    # recompute the scan!
    #print(np.array(root["observations"]["theta_sin"][start:end+1]).shape)

    theta = np.concatenate((np.array(root["observations"]["theta_sin"][start:end+1]).reshape(-1,1),
                            np.array(root["observations"]["theta_cos"][start:end+1]).reshape(-1,1)),axis=1)


    # add dimensions at dim=1
    if use_sim_laser:#
        laser_pos = np.concatenate((xy,theta),axis=1)
        new_laser = F110Env.get_laser_scan(laser_pos,20)
        
        new_laser = F110Env.normalize_laser_scan(new_laser)
        new_laser = np.expand_dims(new_laser,1)
    else:
        new_laser = np.array(root["observations"]["lidar_occupancy"][start:end+1])
    #print(root["observations"]["lidar_occupancy"][start:end+1].shape)
    crash, positions = F110Env.termination_scan(xy,new_laser,
                                           target_speed)
    actual_crash = root["done"][start:end+1].any()
    # set all to true that are larger than the first crash (inclusive)
    first_crash = np.where(crash==1)[0]
    if first_crash.size > 0:
        crash[first_crash[0]+1:] = 1
    new_terminals[start:end+1] = crash
    if crash.any():
        if actual_crash:
            true_positves += 1
            trajectories_true_positives.append(xy)
            trajectory_terminations_true_positives.append(crash)
        else:
            false_positives += 1
            trajectory_false_positive.append(xy)
            trajectory_terminations_false_positive.append(crash)
    else:
        if actual_crash:
            missed += 1
            trajectories_missed.append(xy)
            names_missed.append(model_names[start])
            print(names_missed[-1])
            trajectory_terminations_missed.append(root["done"][start:end+1])
            plot_trajectories_on_map(yaml_path, [xy], [root["done"][start:end+1]])
            print(start,end)
        else:
            uncrashing_trajectories += 1
    
            #print("actual crash not detected")
    # for debugging
    #if i > 100:
    #    break

# root["compute_termination"] = new_terminals | root["done"]
print("missed:", missed)
print(np.unique(np.array(names_missed), return_counts=True))
print(len(np.array(names_missed)))

for name in np.unique(np.array(names_missed)):
    print(name)
    print(np.where(np.array(names_missed)==name)[0])
  
    trajectories_missed = np.array(trajectories_missed)
    trajectory_terminations_missed = np.array(trajectory_terminations_missed)
    plot_trajectories_on_map(yaml_path, trajectories_missed[np.where(np.array(names_missed)==name)[0]], trajectory_terminations_missed[np.where(np.array(names_missed)==name)[0]])
print("missed", missed)
plot_trajectories_on_map(yaml_path, trajectories_missed, trajectory_terminations_missed)

print(sum(new_terminals))
print(new_terminals.shape)
print(root["done"].shape)
print("true_positives:", true_positves)
plot_trajectories_on_map(yaml_path, trajectories_true_positives, trajectory_terminations_true_positives)
print("false positives", false_positives)
plot_trajectories_on_map(yaml_path, trajectory_false_positive, trajectory_terminations_false_positive)

print("still not crashing", uncrashing_trajectories)