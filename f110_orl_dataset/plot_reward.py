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
