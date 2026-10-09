"""Compute a reward for every row of a dataset and store it as new_rewards/<reward_config> in the zarr archive.

Example (adds the rewards that f110-sim-stoch-v2 does not ship with):
    python relabel_reward.py --dataset f110-sim-stoch-v2 --reward_config reward_progress.json

By default the progress along the track is recomputed from the poses, as for the ground truth of the OPE benchmark.
The new_rewards shipped with f110-real-stoch-v2 were computed from the stored progress observation instead,
use --stored_progress to reproduce them.
"""
import argparse
from pathlib import Path
import gymnasium as gym
import zarr
import numpy as np
import f110_gym
import f110_orl_dataset

parser = argparse.ArgumentParser(description='Relabel the rewards of an f110 dataset')
parser.add_argument('--dataset', type=str, default="f110-real-stoch-v2", choices=["f110-real-stoch-v2", "f110-sim-stoch-v2"])
parser.add_argument('--reward_config', type=str, default="reward_progress.json", help="reward config file")
parser.add_argument('--path', type=str, default=None, help="zarr directory, defaults to ~/.f110_rl_datasets/<dataset>")
parser.add_argument('--map', type=str, default="Infsaal3", help="map name")
parser.add_argument('--stored_progress', action='store_true', help="use the stored progress observation (not in f110-sim-stoch-v2)")


if __name__ == "__main__":
    args = parser.parse_args()
    location = dict(data_dir=Path(args.path), skip_download=True) if args.path is not None else {}
    # same settings as experiments/compute_reward_from_ds.py of the OPE benchmark, but with all filters disabled
    # so that the loaded rows are exactly the rows of the zarr archive
    F110Env = gym.make(args.dataset,
        **location,
        encode_cyclic=not args.stored_progress, # the stored progress is only available as "progress" without cyclic encoding
        flatten_obs=True,
        use_delta_actions=True,
        delta_factor=1.0,
        include_time_obs=False,
        include_progress=args.stored_progress,
        use_compute_termination=True,
        set_terminals=True,
        remove_cons_terminals=False,
        include_vesc_fault_trajectories=True,
        bad_trajectories=[],
        bad_agents=[],
        **dict(name=args.dataset,
            config = dict(map=args.map, num_agents=1, params=dict(vmin=0.0, vmax=2.0)),
            render_mode="human")
    ).unwrapped
    dataset = F110Env.get_dataset()
    root = zarr.open(str(F110Env.data_dir), mode='r+')
    num_rows = len(root["done"])
    assert len(dataset["observations"]) == num_rows, "loaded rows do not match the zarr archive"

    # episodes are delimited by the recorded done/truncated flags; like the benchmark (set_terminals=True) an episode
    # terminates at its first compute_termination or at its truncation
    ends = np.where(root["done"][:] | root["truncated"][:])[0]
    starts = np.r_[0, ends[:-1] + 1]
    termination = root["compute_termination"][:] | root["truncated"][:]
    horizon = np.max(ends + 1 - starts)
    trajectories = np.zeros((len(starts), horizon, dataset["observations"].shape[1]))
    actions = np.zeros((len(starts), horizon, dataset["actions"].shape[1]))
    terminations = np.full(len(starts), horizon + 1)
    for i, (start, end) in enumerate(zip(starts, ends)):
        trajectories[i, :end - start + 1] = dataset["observations"][start:end + 1]
        actions[i, :end - start + 1] = dataset["actions"][start:end + 1]
        term = np.where(termination[start:end + 1])[0]
        if len(term) > 0:
            terminations[i] = term[0]

    reward = F110Env.compute_reward_trajectories(trajectories, actions, terminations, args.reward_config,
                                                 precomputed_progress=args.stored_progress)
    new_reward = np.zeros(num_rows)
    for i, (start, end) in enumerate(zip(starts, ends)):
        new_reward[start:end + 1] = reward[i, :end - start + 1]

    root.require_group("new_rewards").array(args.reward_config, new_reward, overwrite=True)
    print(f"Finished relabeling, available as {args.reward_config} in new_rewards group")
