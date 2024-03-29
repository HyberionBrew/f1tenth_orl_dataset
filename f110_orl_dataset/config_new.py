import json
import os
class RewardConfig:
    def __init__(self, filepath):
        # find the current path and append the config file name
        self.filepath = os.path.join(os.path.dirname(__file__), filepath) 
        #self.filepath = filepath
        self.load_config()

    def load_config(self):
        with open(self.filepath, 'r') as file:
            data = json.load(file)
            self.progress_reward_weight = data.get('progress_reward_weight', 0.0)
            self.min_action_weight = data.get('min_action_weight', 0.0)
            self.min_lidar_weight = data.get('min_lidar_weight', 0.0)
            self.raceline_delta_weight = data.get('raceline_delta_weight', 0.0)
            self.min_steering_weight = data.get('min_steering_weight', 0.0)
            self.sparse_reward = data.get('sparse_reward', False)
            self.collision_penalty = data.get('collision_penalty', 0.0)
            self.lifetime_weight = data.get('lifetime_weight', 0.0)
            self.checkpoint_reward_weight = data.get('checkpoint_reward_weight', 0.0)
            # self.progress_reward_multiplier = data.get('progress_reward_multiplier', 1)

    def has_progress_reward(self):
        return self.progress_reward_weight > 1e-6  # A small number close to zero
    def has_lifetime_reward(self):
        return self.lifetime_weight > 1e-6
    
    def has_min_action_reward(self):
        return self.min_action_weight > 1e-6
    def has_min_lidar_reward(self):
        return self.min_lidar_weight > 1e-6
    def has_raceline_delta_reward(self):
        return self.raceline_delta_weight > 1e-6
    def has_checkpoint_reward(self):
        return self.checkpoint_reward_weight > 1e-6
    def has_sparse_reward(self):
        return self.sparse_reward
    
    # not implemented yet!
    def has_min_steering_reward(self):
        return self.min_steering_weight > 1e-6

    def __str__(self):
        return f'Config(progress_reward_weight={self.progress_reward_weight}, min_action_weight={self.min_action_weight}, min_lidar_weight={self.min_lidar_weight}, raceline_delta_weight={self.raceline_delta_weight}, min_steering_weight={self.min_steering_weight}, collision_penalty={self.collision_penalty}, lifetime_weight={self.lifetime_weight}, sparse_reward={self.sparse_reward})'