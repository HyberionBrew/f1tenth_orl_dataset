import argparse
from f110_orl_dataset.config_new import RewardConfig
import gymnasium as gym
import zarr
from f110_orl_dataset import fast_reward
import numpy as np

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--reward_config', type=str, default="reward_config.json", help="reward config file")
parser.add_argument('--path', type=str, default="datasets", help="dataset name")
parser.add_argument('--map', type=str, default="Infsaal2", help="map name")
args = parser.parse_args()


if __name__ == "__main__":
    
    #reward_config = RewardConfig(args.reward_config)
    #print("Reward config", reward_config)
    zarr_path = args.path
    F110Env = gym.make('f110-real-v0',
    # only terminals are available as of tight now 
        encode_cyclic = False,
        flatten_obs=True,
        remove_cons_terminals=False,
        use_compute_termination=True,
        set_terminals=False,
        **dict(name='f110-real-v0',
            config = dict(map=args.map, num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
            zarr_path= args.path,
            #clip_trajectory_length=(0,timestep),
            )
    print(F110Env.get_observation_keys())
    print(dataset["observations"].shape)
    trajectories, action_trajectories, terminations, model_names = F110Env.compute_trajectories(dataset["observations"],
                                                                                            dataset["actions"],
                                                                                            dataset["terminals"],
                                                                                            dataset["timeouts"], 
                                                                                            dataset["model_name"])
 
    print("Trajectories", trajectories.shape)
    print("Action trajectories", action_trajectories.shape)
    print("Terminations", terminations.shape)
    print("Model names", model_names.shape)
    print("0 length", np.sum(terminations==0))

    reward = F110Env.compute_reward_trajectories(trajectories, 
                                    action_trajectories,
                                     terminations,
                                     args.reward_config,
                                     precomputed_progress=True)

    import matplotlib.pyplot as plt
    #plt.plot(terminations[:1000])
    #plt.show()
    #plt.plot(reward[132])
    #plt.plot(reward[0])
    #plt.show()
    # now we need to correctly assing the rewards back to the normal dataset
    from tqdm import tqdm
    new_reward = np.zeros((dataset["observations"].shape[0]))

    touched = np.zeros((dataset["observations"].shape[0]))
    start = 0
    for trajectory_reward, termination in tqdm(zip(reward, terminations), total=len(reward)):
        #print(termination)
        if termination > trajectories.shape[1]:
            termination = trajectories.shape[1]
        else:
            termination = int(termination) + 1
        end = start + termination
        #print(trajectory_reward.shape)
        #print(new_reward[start:end+1].shape)
        new_reward[start:end] = trajectory_reward[0:termination]
        touched[start:end] = 1
        start = end
        #print(end)
        #print(termination)
        #print(len(new_reward[start:end]))
    root = zarr.open(zarr_path, mode='wr') 
    root["new_rewards"][args.reward_config] = new_reward
    print(f"Finished relabeling, available as {args.reward_config} in new_rewards group")
    """
    
    # check if group exists
    if "new_rewards" in root:
        new_rewards_group = root["new_rewards"]
    else:
        new_rewards_group = root.create_group("new_rewards")
    # new_rewards_group.create_dataset("lidar_reward", shape=shape, dtype=dtype)
    if reward_config.has_sparse_reward():
        new_rewards = fast_reward.sparse_reward(dataset)
        root["new_rewards"][args.reward_config] = new_rewards
    else:
        print(dataset["observations"].shape)
        new_rewards = fast_reward.calculate_reward(reward_config, dataset, F110Env, F110Env.track)
        # new_rewards *= 10.0
        print(new_rewards.shape)
        new_rewards = np.squeeze(new_rewards, axis=0)

        root["new_rewards"][args.reward_config] = new_rewards
    # where there is a terminal set reward to 0
    #print("Number of 0s", np.count_nonzero(new_rewards == 0))
    print(root["done"].shape)
    print(new_rewards.shape)
    #new_rewards[root["done"]] = 0.0
    print(root["new_rewards"][args.reward_config][:150])
    # count 0s in new rewards
    print(len(root["new_rewards"][args.reward_config]))
    
    #print("Number of 0s", np.count_nonzero(new_rewards == 0))
    #import matplotlib.pyplot as plt
    #for i in range(10):
    #    plt.plot(np.array(new_rewards[0:2500 ]))
        #plt.plot(np.array(root["truncated"][500 * i:500 * (i+1)]))
        #plt.plot(np.array(root["done"][500 * i:500 * (i+1)]))
    #     plt.show()
    print("Finished relabeling, now available as --alternate_rewards=True")
    """