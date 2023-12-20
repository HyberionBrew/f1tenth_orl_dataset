import argparse
from f110_orl_dataset.config_new import Config as RewardConfig
import gymnasium as gym
import zarr
from f110_orl_dataset import fast_reward
import numpy as np

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--reward_config', type=str, default="reward_config.json", help="reward config file")
parser.add_argument('--path', type=str, default="datasets", help="dataset name")
args = parser.parse_args()


if __name__ == "__main__":
    
    reward_config = RewardConfig(args.reward_config)
    print("Reward config", reward_config)
    zarr_path = args.path
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
            zarr_path= args.path, 
            alternate_reward=False,
            include_timesteps_in_obs=True,
            only_terminals=False,
            debug=False,
            #clip_trajectory_length=(0,timestep),
            )
    
    root = zarr.open(zarr_path, mode='wr')
    if reward_config.has_sparse_reward():
        new_rewards = fast_reward.sparse_reward(dataset)
        root['new_reward'] = new_rewards
    else:
        new_rewards = fast_reward.calculate_reward(reward_config, dataset, F110Env, F110Env.track)
        new_rewards *= 10.0
        new_rewards = np.squeeze(new_rewards, axis=0)

        root['new_rewards'] = new_rewards
    print("Finished relabeling, now available as --alternate_rewards=True")