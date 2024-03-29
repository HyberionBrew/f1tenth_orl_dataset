import numpy as np

from f110_gym.envs.track import Track
import gymnasium as gym
from f110_orl_dataset.normalize_dataset import Normalize
import warnings

class ProgressReward:
    def __init__(self, multiplier=100.0):
        self.multiplier = multiplier
    """
    (Batch, trajectory_length, obs_dim/act_dim) -> input
    """
    def __call__(self, obs, action, laser_scan):
        #if obs.shape[1] <= 1:
        #    return np.zeros((obs.shape[0], obs.shape[1]))
        #assert len(obs.shape) == 3
        #assert obs.shape[1] > 1
        if isinstance(obs, dict):
            progress = obs["progress"].flatten()
            # add a zero dimension
            progress = np.expand_dims(progress, axis=0)
            #print(progress)
        else: 
            progress = obs [..., -2]
        # assert progress.shape[-1] == 2
        # sin and cos progress to progress
        #progress = np.arctan2(progress[...,0], progress[...,1])
        # all where progress < 0 we add pi
        #progress += np.pi
        #progress = progress/ (2*np.pi)
        #print(progress.min())
        #print(progress.max())
        assert (progress.min() >= 0) and (progress.max() <= 1)
        #print(progress)
        #print(progress.shape)
        # along the second dimension we take the diff
        delta_progress = np.diff(progress, axis=1)
        # need to handle the case where we cross the 0/1 boundary
        # find indices where delta_progress < -0.9
        indices = np.where(delta_progress < -0.9)
        # at these indices we need to add 1 to delta_progress
        delta_progress[indices] += 1
        delta_progress *= self.multiplier
        # max between 0 and delta_progress
        delta_progress = np.maximum(0, delta_progress)
        reward = delta_progress
        # append a zero to the reward
        reward = np.concatenate([reward,np.zeros_like(reward[...,0:1])], axis=-1)
        return reward

class MinActReward:
    def __init__(self,low_steering, high_steering):
        self.low_steering = low_steering
        self.high_steering = high_steering
    """
    @param obs: observation with the shape (batch_size, trajectory_length, obs_dim).
    @returns reward with the shape (batch_size, trajectory_length)
    """
    def __call__(self, obs, action, laser_scan):
        #raise NotImplementedError # TODO! this has to be adapted
        # higher reward for smaller steering
        # get the previous steering and add the action
        #print(obs.keys())
        prev_steer = obs["previous_action_steer"]

        
        raw_steering = prev_steer + action[:,0]
        #print(raw_steering.shape)
        delta_steering = np.abs(raw_steering)
        normalized_steering = (delta_steering / self.high_steering)**2
        inverse = 1 - normalized_steering
        reward = inverse
        """
        import matplotlib.pyplot as plt
        plt.plot(reward)
        plt.plot(raw_steering)
        plt.legend(["Reward", "Steering"])
        plt.show()
        """
        return np.expand_dims(reward, axis=0)


# harder to test right now, so implementing & testing later
class MinLidarReward:
    def __init__(self, high=0.15):
        self.high = high
    def __call__(self,obs,action, laser_scan):
        #print("called min lidar reward")
        #print("laser scan shape", laser_scan.shape)
        laser_scan = laser_scan[...,:]
        # remove all one dimensions
        laser_scan= np.squeeze(laser_scan)
        # sort along axis 1
        laser_scan = np.sort(laser_scan, axis=1) 
        # take the 3d smallest value
        min_lidar = laser_scan[...,2]
        # normalize between 0 and 1
        # first clip
        min_lidar = np.clip(min_lidar, 0, self.high)
        # normalize
        min_lidar = min_lidar / self.high
        min_ray = min_lidar **4
        #print(min_ray.shape)
        #print(min_ray)
        reward = min_ray
        #print("reward scan shape", reward.shape)
        # add a 1 dimension at the start
        reward = np.expand_dims(reward, axis=0)
        # assert reward.shape == (obs.shape[0], obs.shape[1])
        return reward

class LifetimeReward:
    def __init__(self):
        pass
    def __call__(self, obs, action, laser_scan):
        if isinstance(obs, dict):
            reward = np.ones((1, obs["poses_x"].shape[0]))
        else:
            reward = np.ones((obs.shape[0], obs.shape[1]))
        return reward

class RacelineDeltaReward:
    def __init__(self, track:Track, max_delta=2.0):
        xs = track.raceline.xs
        ys = track.raceline.ys
        #print(xs[0])
        #print(ys[0])
        self.raceline = np.stack([xs,ys], axis=1)
        self.largest_delta_observed = max_delta
        #print(track)
        #print(self.raceline)
        #print(self.raceline.shape)

    def __call__(self, obs, action, laser_scan):
        # check if obs is a dictionary
        if isinstance(obs, dict):
            pose = np.concat((obs["poses_x"], obs["poses_y"]), axis=-1) 
        else:
            pose = obs[...,:2]
        # batch_data shape becomes (batch_size, timesteps, 1, 2)
        # racing_line shape becomes (1, 1, points, 2)
        pose_exp = np.expand_dims(pose, axis=2)
        racing_line_exp = np.expand_dims(self.raceline, axis=0)
        racing_line_exp = np.expand_dims(racing_line_exp, axis=0)
        #distances = np.sum((racing_line_exp - np.array(pose_exp))**2, axis=1)
        #print(distances.shape)
        #min_distance_squared = np.min(distances,axis=1)
        squared_distances = np.sum((pose_exp - racing_line_exp) ** 2, axis=-1)
        # print(min_distance_squared.shape)
        min_distances = np.sqrt(np.min(squared_distances, axis=-1, keepdims=True))
        #clip reward to be between 0 and largest_delta
        min_distances = np.clip(min_distances, 0, self.largest_delta_observed)
        min_distance_norm = min_distances / self.largest_delta_observed
        reward = 1 - min_distance_norm
        reward = reward **2
        # print(reward.shape)
        # remove last dimension
        reward = reward[...,0]
        # plot the poses
        if False:
            import matplotlib.pyplot as plt
            plt.scatter(pose[...,0], pose[...,1], color='red')
            plt.scatter(self.raceline[...,0], self.raceline[...,1], color='blue')
            plt.show()
        return reward

class CheckpointReward:
    def __init__(self, num_checkpoints: int=10):
        self.num_checkpoints = num_checkpoints
        self.checkpoints = np.linspace(0,1,num_checkpoints+1)
    def __call__(self, obs, action, laser_scan):
        # flatten obs progress
        progress = obs["progress"].flatten()
        # progress = obs["progress"][:]
        #print(progress.shape)
        # give reward if we pass a checkpoint (once)
        differences = np.diff(progress)
        loop_back_indices = np.where(differences < -0.5)[0]
        for idx in loop_back_indices:
            progress[idx+1:] += 1.0
        #import matplotlib.pyplot as plt
        #plt.plot(progress)
        #plt.show()
        #print(loop_back_indices)
        #print(progress.shape)
        # return array with ones where we pass a checkpoint
        smallest = progress[0]
        first_checkpoint = np.where(self.checkpoints > smallest)[0][0]
        checkpoint_list = self.checkpoints[first_checkpoint] + self.checkpoints
        latest_checkpoint = checkpoint_list[-1]
        #print(checkpoint_list)
        progress[-1] = 3
        while latest_checkpoint < progress[-1]:
            # concatenate to checkpoint list checkpoint_list + self.checkpoints
            checkpoint_list = np.concatenate([checkpoint_list, checkpoint_list[-1] + self.checkpoints])
            #print(checkpoint_list)
            latest_checkpoint = checkpoint_list[-1]
        rewards = np.zeros((1,progress.shape[0]))
        for checkpoint in checkpoint_list:
            larger = np.where(progress > checkpoint)[0]
            if len(larger) == 0:
                break
            else:
                rewards[0,larger[0]] += 1.0
        #plt.plot(rewards[0])
        #plt.plot(progress)
        #plt.show()
        return rewards

# CURRENTLY DOES NOT WORK WITH LIDAR!
#TODO! CAREFULL WHEN ADDING LIDAR TO NOT BREAK SOME REWARDS
class MixedReward:
    def __init__(self, env:gym.Env, config):
        # self.config = config
        self.env = env
        self.rewards = []
        self.config = config
        self.add_rewards_based_on_config(config)

    def add_rewards_based_on_config(self,config):
        self.rewards = []
        if config.has_progress_reward():
            self.rewards.append((ProgressReward(),config.progress_reward_weight))
        if config.has_min_action_reward():
            self.rewards.append((MinActReward(self.env.action_space.low[0][0],
                                            self.env.action_space.high[0][0]), config.min_action_weight))
        if config.has_min_lidar_reward():
            self.rewards.append((MinLidarReward(), config.min_lidar_weight))
        if config.has_raceline_delta_reward():
            print("track length", len(self.env.track.centerline.xs))
            print("track length", len(self.env.track.raceline.xs))
            self.rewards.append((RacelineDeltaReward(self.env.track),config.raceline_delta_weight))
        if config.has_lifetime_reward():
            self.rewards.append((LifetimeReward(), config.lifetime_weight))
        if config.has_checkpoint_reward():
            self.rewards.append((CheckpointReward(10), config.checkpoint_reward_weight))
        if config.has_min_steering_reward():
            pass
            #self.rewards.append(MinSteeringReward())
    """
    @param obs: observation with the shape (batch_size, trajectory_length, obs_dim).
    Trajectory length needs to be at least > 1 for certain rewards.
    @param action: action with the shape (batch_size, trajectory_length, action_dim)
    """
    # mixedReward(batch[0], batch[1], batch[2], batch[3], laser_scan=batch[4])
    def __call__(self, obs, action, collision, done, laser_scan=None):
        # print a deprecation warning that states obs should be dict


        warnings.warn("Reward computation with flattened obs is deprecated. Please use the unflatten function to convert the flattened obs to a dict.")
        assert obs.shape[:-1] == action.shape[:-1], f" Obs shape is {obs.shape} and action shape is {action.shape}"
        

        assert len(obs.shape) == 3
        #print(obs.shape)
        # assert obs.shape[-1] == 7
        # need to handle laser scans somehow in the future

        # empty rewards array to collect the rewards
        rewards = np.zeros((obs.shape[0], obs.shape[1]))
        # now we need to handle each of the rewards
        #print("***", obs.shape)
        for reward in self.rewards:

            rewards += reward(obs, action, laser_scan)
        #print(rewards.shape)
        # where collision is true set the reward to -10
        rewards[collision:] = self.config.collision_penalty

        return rewards, None
    
    def __call__(self, obs:dict, action, collision, scan):
        """
        Takes in a single trajectory and computes the reward for it.

        Args:
            obs (dict): _description_
            action (_type_): _description_
            collision (_type_): _description_
            scan (_type_): _description_

        Returns:
            _type_: _description_
        """
        rewards = np.zeros((1, obs["poses_x"].shape[0]))
        # now we need to handle each of the rewards
        #print("***", obs.shape)
        for reward, weight in self.rewards:
            r = reward(obs, action, scan)
            rewards += r * weight
        #print(rewards.shape)
        # where collision is true set the reward to -10
        rewards[0,collision:] = self.config.collision_penalty
        rewards[0,collision+1:] = 0.0 # set rest to 0.0
        #print(rewards)
        #print(rewards.shape)
   
        #import matplotlib.pyplot as plt
        #plt.plot(rewards[0])
        #plt.title("Rewards")
        #plt.show()
        #exit()
        return rewards, None

class StepMixedReward:
    def __init__(self, env, config):
        self.mixedReward = MixedReward(env, config)
        self.previous_obs = None
        self.previous_action = None
    def reset(self):
        self.previous_obs = None
        self.previous_action = None
    """
    obs need to have the following shape: (batch, 1, obs_dim) (since we only do stepwise)
    """
    def __call__(self, obs, action, collision, done, laser_scan=None):
        # print(obs)
        assert len(obs.shape) == 2
        assert len(action.shape) == 2
        assert action.shape[1] == 2
        assert obs.shape[1] == 11
        assert obs.shape[0] == action.shape[0]

        #print(collision.shape)
        #print(done.shape)
        # add a timestep dimension at axis 1
        obs = np.expand_dims(obs, axis=1)
        action = np.expand_dims(action, axis=1)
        collision = np.expand_dims(collision, axis=1)
        done = np.expand_dims(done, axis=1)

        if self.previous_obs is None:
            self.previous_obs = obs
            self.previous_action = action
        
        # now we join the previous obs and action with the current one along dim 1
        obs_t2 = np.concatenate([self.previous_obs, obs], axis=1)
        action_t2 = np.concatenate([self.previous_action, action], axis=1)
        #print(collision)
        collision = np.concatenate([collision, collision], axis=1)
        done = np.concatenate([done, done], axis=1)
        # now we can apply the mixed reward
        reward, _ = self.mixedReward(obs_t2, action_t2, collision, done, laser_scan=laser_scan)
        # now we discard the first timestep
        #print(reward)
        #print(reward.shape)
        reward = reward[:,1]
        self.previous_action = action
        self.previous_obs = obs
        assert reward.shape == (obs.shape[0],) # we only have a batch dimension
        # print(reward)
        return reward



import gymnasium as gym
import f110_orl_dataset


def calculate_reward(config, dataset, env, track):
    # init reward
    mixedReward = MixedReward(env, config)

    timesteps = dataset["observations"].shape[0]
    finished_trajectory = dataset["timeouts"] #np.logical_or(dataset["terminals"],
                          #              dataset["timeouts"])
    
    # find where the trajectory is finished
    finished = np.where(finished_trajectory)[0]

    timesteps = None # for debugging

    #print(dataset["rewards"].shape)
    batch_obs = np.split(dataset["observations"][:timesteps], finished+1)
    batch_act = np.split(dataset["actions"][:timesteps], finished+1)
    batch_col = np.split(dataset["terminals"][:timesteps], finished+1)
    batch_ter = np.split(dataset["terminals"][:timesteps], finished+1)
    batch_laserscan = np.split(dataset["scans"][:timesteps], finished+1)
    #print(batch_laserscan.shape)
    #print(batch_obs.shape)
    all_rewards = np.zeros((1,0))


    
    for batch in zip(batch_obs, batch_act, batch_col, batch_ter, batch_laserscan):
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = np.expand_dims(batch[i], axis=0)
        if batch[0].shape[1] == 1: # trajectory length is 1, just terminals after each other
            reward = np.zeros((1,1))
        elif batch[0].shape[1] == 0:
            continue
        else:
            print("??")
            print(mixedReward(batch[0], batch[1], batch[2], batch[3]))
            reward, _ = mixedReward(batch[0], batch[1], batch[2], batch[3], laser_scan=batch[4])
        #print(reward.shape)
        all_rewards = np.concatenate([all_rewards, reward], axis=1)
    #print(all_rewards[:50])
    return all_rewards

def test_progress_reward():
    from .config_new import Config
    config = Config('reward_config.json')

    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr", 
                alternate_reward=False,
                include_timesteps_in_obs=True,
                only_terminals=False,
                debug=True,
                #clip_trajectory_length=(0,timestep),
                )
    # apply the reward
    all_rewards = calculate_reward(config, dataset)
    rewards = dataset["rewards"]
    # set start of trajectory to zero, because this is 
    # different when computed offline
    finished_trajectory = np.logical_or(dataset["terminals"],
                                    dataset["timeouts"])
    finished = np.where(finished_trajectory)[0]
    rewards[0] = 0
    # add a zero at the end of rewards so we dont go ooB
    rewards = np.concatenate([rewards, np.zeros(1)])
    rewards[finished+1] = 0
    # remove the zero
    rewards = rewards[:-2]
    all_rewards = all_rewards[0,:rewards.shape[0]]
    indices = np.where(np.isclose(all_rewards, rewards, atol = 1e-5) == False)[0]
    # at these indices print the difference between the two
    #print(indices[:3])
    #print("Difference")
    #print(all_rewards[indices]- rewards[indices])
    assert len(indices) == 0
    print("[td_progress] Test passed")


def create_batch(obs_steps_t, action_steps_t, done_t, truncate_t, i):
    obs_steps = np.expand_dims(obs_steps_t[i], axis=[0,1])
    action_steps = np.expand_dims(action_steps_t[i], axis=[0,1])
    
    # make the batch size 2
    obs_steps = np.concatenate([obs_steps, obs_steps], axis=0)
    action_steps = np.concatenate([action_steps, action_steps], axis=0)
    # same for done truncate
    # expand done and truncate dimensions
    done = np.expand_dims(done_t[i], axis=[0,1])
    truncate = np.expand_dims(truncate_t[i], axis=[0,1])
    done = np.concatenate([done, done], axis=0)
    truncate = np.concatenate([truncate, truncate], axis=0)
    return obs_steps, action_steps, done, truncate

def test_online_batch_reward():
    from .config_new import Config
    config = Config('reward_config.json')
    stepMixedReward = StepMixedReward(config)

    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr", 
                alternate_reward=False,
                include_timesteps_in_obs=True,
                only_terminals=False,
                #clip_trajectory_length=(0,timestep),
                )
    
    horizon = 1000000
    batch_size = 2
    # create a batch with batch_size trajectories taken from the dataset
    obs_steps_t = dataset["observations"][:horizon]
    action_steps_t = dataset["actions"][:horizon]
    
    done_steps_t = np.logical_or(dataset["terminals"][:horizon],
                                dataset["timeouts"][:horizon])
    collision_steps_t = dataset["terminals"][:horizon]

    rewards = np.zeros((2, horizon))
    for i in range(horizon):
        obs, act, done, coll = create_batch(obs_steps_t, 
                                            action_steps_t, 
                                            done_steps_t, 
                                            collision_steps_t,
                                            i)
        #print(done.shape)
        #print(coll.shape)
        # print(done.any())
        #print(obs.shape)
        reward = stepMixedReward(obs, act, coll, done)
        if done.any():
            print("Done", i)
            stepMixedReward.reset()
        #print(reward.shape)
        #print(reward)
        rewards[:,i] = reward
    
    precompted_rewards = dataset["rewards"][:horizon]

    indices = np.where(np.isclose(precompted_rewards, rewards[0], atol = 1e-5) == False)[0]
    print(indices)
    print()
    print(rewards[0].shape)
    print(precompted_rewards.shape)
    
    #print(indices)
    print(rewards[0])
    print(rewards[1])
    
    print(dataset["rewards"][:10])

    print(rewards[0][1000])
    print(dataset["rewards"][1000])
    print(precompted_rewards[indices])
    print(rewards[0][indices])
    # apply the reward in batch

def test_reward(config_file, dataset_folder):
    from .config_new import Config
    config = Config(config_file)
    
    # stepMixedReward = StepMixedReward(config)

    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{dataset_folder}", 
                alternate_reward=True,
                include_timesteps_in_obs=True,
                only_terminals=False,
                )
    normalizer = Normalize()
    print(config)
    all_rewards = calculate_reward(config, dataset, F110Env, F110Env.track)

    rewards = dataset["rewards"]
    # set start of trajectory to zero, because this is 
    # different when computed offline
    finished_trajectory = np.logical_or(dataset["terminals"],
                                    dataset["timeouts"])
    finished = np.where(finished_trajectory)[0]
    # rewards[0] = 0
    # add a zero at the end of rewards so we dont go ooB
    rewards = np.concatenate([rewards, np.zeros(1)])
    # rewards[finished+1] = 0
    # remove the zero
    rewards = rewards[:-2]
    all_rewards = all_rewards[0,:rewards.shape[0]]
    indices = np.where(np.isclose(all_rewards, rewards, atol = 1e-5) == False)[0]
    # at these indices print the difference between the two
    #print(indices[:3])
    #print("Difference")
    #print(all_rewards[indices]- rewards[indices])
    print(indices)
    print("Computed 10:", all_rewards[:10])
    print("Ground truth 10:", rewards[:10])
    print("Actions 10:", dataset["raw_actions"][:10])
    print(all_rewards[10063])
    print(rewards[10063])
    print(dataset["raw_actions"][10063])
    assert len(indices) == 0
    print(f"[{dataset_folder}] Test passed")

class RewardCalculator:
    def __init__(self,config, scale = 10.0):
        self.config = config
        self.scale = scale
    def forward(self, ):
        pass
        

def reward_from_config(config_file, dataset_folder, scale = 10.0):
    if reward_config.has_sparse_reward():
        new_rewards = fast_reward.sparse_reward(dataset)
        root['new_reward'] = new_rewards
    else:
        new_rewards = fast_reward.calculate_reward(reward_config, dataset, F110Env, F110Env.track)
        new_rewards *= scale
        new_rewards = np.squeeze(new_rewards, axis=0)


def sparse_reward(dataset):
    progress = dataset["observations"][:,-2:]
    # calculate the progress
    print(progress.shape)
    progress = np.arctan2(progress[...,0], progress[...,1])
    # from [-pi, pi] to [0, 1]
    progress += np.pi
    progress = progress / (2*np.pi)
    # or truncated and dones
    finished = np.logical_or(dataset["terminals"], dataset["timeouts"])
    starts = np.roll(finished, 1)

    starts = np.where(starts==1)[0]
    ends = np.where(finished==1)[0]
    rewards = np.zeros_like(progress)
    print(rewards.shape)
    print("...")
    for start,end in zip(starts,ends):
        # find all loopbacks
        #print(start, end)
        progress_slice = progress[start:end+1].copy()
        rewards_slice = rewards[start:end+1].copy()
        differences = np.diff(progress[start:end+1])
        # find where there is a diff of larger than 0.5, in these cases we have a loopback
        # and must add 1 to all subsequent progress values
        # find where differences < -0.5
        indices = np.where(differences < -0.5)[0]
        #print(indices)
        #print(differences)
        #plt.plot(progress_slice)
        #plt.show()
        # add 1 to all subsequent progress values
        for index in indices[::-1]:
            #print(progress_slice[index])
            #print(progress_slice[index+1])
            progress_slice[index+1:] += progress_slice[index]
            #print(progress_slice[index])
            #print(progress_slice[index+1])
            #print(index)
        without_offset = progress_slice- progress_slice[0]
        max_threshold = np.floor(without_offset[-1] * 10) / 10

        # Generate an array of thresholds to check
        thresholds = np.arange(0.1, max_threshold + 0.1, 0.1)
        # print(thresholds)
        # Find the first index where the array value exceeds each threshold
        first_exceedances = np.array([np.argmax(without_offset >= threshold) for threshold in thresholds])
        #print(first_exceedances)
        #print(first_exceedances)
        if len(first_exceedances) > 0:
            rewards_slice[first_exceedances] = 10.0
            rewards[start:end+1] = rewards_slice
    #import matplotlib.pyplot as plt
    #plt.plot(rewards)
    #plt.show()
    return rewards
def test_sparse_reward(dataset_folder):
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal2", num_agents=1),
            render_mode="human")
    )

    dataset =  F110Env.get_dataset(
                zarr_path= f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{dataset_folder}", 
                alternate_reward=False,
                include_timesteps_in_obs=True,
                only_terminals=False,
                # clip_trajectory_length =(0,500),
                )
    reward = sparse_reward(dataset)
    print(reward)
    # create histogram of number of ones in reward, bin size 1000
    import matplotlib.pyplot as plt
    chunk_size = 10000
    chunk_sums = [np.sum(reward[i:i + chunk_size]) for i in range(0, len(reward), chunk_size)]

    # Create the plot
    plt.plot(chunk_sums)
    plt.xlabel('Time (in chunks of 1000 timesteps)')
    plt.ylabel('Sum of values per chunk')
    plt.title('Aggregated Timeseries Data')
    plt.show()

import zarr
if __name__ == "__main__":
    dataset_folder = "trajectories_1112.zarr"
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )
    zarr_path = f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/{dataset_folder}"
    dataset =  F110Env.get_dataset(
                zarr_path= zarr_path, 
                alternate_reward=False,
                include_timesteps_in_obs=True,
                only_terminals=False,
                #clip_trajectory_length =,#(0,500),
                )
    new_rewards = sparse_reward(dataset)
    
    root = zarr.open(zarr_path, mode='wr')
    root["new_rewards"] = new_rewards
    print(root["new_rewards"].shape)
    print(root["rewards"].shape)
    reward = root["new_rewards"]

    import matplotlib.pyplot as plt
    chunk_size = 10000
    chunk_sums = [np.sum(reward[i:i + chunk_size]) for i in range(0, len(reward), chunk_size)]

    # Create the plot
    plt.plot(chunk_sums)
    plt.xlabel('Time (in chunks of 1000 timesteps)')
    plt.ylabel('Sum of values per chunk')
    plt.title('Aggregated Timeseries Data')
    plt.show()
    #test_progress_reward()
    # test_online_batch_reward()
    #test_reward("reward_min_act.json", "trajectories_min_act.zarr")
    # test_reward("reward_raceline.json", "trajectories_raceline.zarr")