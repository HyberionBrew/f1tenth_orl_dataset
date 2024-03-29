import numpy as np


def calculate_discounted_reward(dataset, gamma=0.99):
    unique_models = np.unique(dataset['model_name'])
    start_points = np.where(np.roll(dataset['timeouts'],1))[0]
    end_points = np.where(dataset['timeouts'])[0]
    rewards = dataset["rewards"]# [reward_config]
    rewards_dict = {}
    for model in unique_models:
        model_mask = np.where(dataset['model_name'] == model)[0]
        # get the start and end points:
        model_start_points = np.intersect1d(model_mask, start_points)
        model_end_points = np.intersect1d(model_mask, end_points)
        model_discounted_rewards = []
        for start, end in zip(model_start_points, model_end_points):
            segment_rewards = rewards[start:end + 1]
            # print(len(segment_rewards))
            discounted_reward = np.sum(segment_rewards * gamma ** np.arange(len(segment_rewards)))
            model_discounted_rewards.append(discounted_reward)
        model_discounted_rewards = np.array(model_discounted_rewards)
        rewards_dict[model] = {}
        rewards_dict[model]["mean"] = np.mean(model_discounted_rewards)
        rewards_dict[model]["std"] = np.std(model_discounted_rewards)

    return rewards_dict

def plot_rewards(dataset,reward_config, rewards_dict):
    import plot_utilities as pu
    rewards = calculate_discounted_reward(dataset,reward_config)
    target = f"reward-{reward_config}"
    keys = np.unique(dataset['model_name'])
    # print(rewards)
    all_rewards = {"ground_truth":{target:{'250':rewards}}}
    pu.plot_bars_from_dict(all_rewards, 
                        target=target, 
                        length='250', 
                        methods= ["ground_truth"],#, "fqe", "dr"],
                        sub_keys=keys,
                        add_title = "; raceline",
                        path="test.png")
    
if __name__ == "__main__":
    import gymnasium as gym

    F110Env = gym.make('f110-sim-v1',
    # only terminals are available as of tight now 
                       clip_trajectory_length=(0,250),
                       et_previous_step_terminals=25,
                       reward_config="reward_progress.json",
        **dict(name='f110-sim-v1',
            config = dict(map="Infsaal2", num_agents=1),
            render_mode="human")
    )
    zarr_path = f"/home/fabian/msc/f110_dope/ws_release/real_ds_127.zarr"
    dataset =  F110Env.get_dataset(
                zarr_path= zarr_path, 
                
                # only_agents = ["pure_pursuit2_0.8_1.2_raceline_og_3_0.6"],
                #clip_trajectory_length =,#(0,500),
                )
    reward_config = "reward_progress.json"
    rewards_dict = calculate_discounted_reward(dataset,reward_config)
    print(rewards_dict)
    plot_rewards(dataset,reward_config, rewards_dict)