import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment


if __name__ == "__main__":

    seeds = [42, 123, 456, 789, 1000]

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Running baseline with seed: {seed}")
        print(f"{'='*40}")

        env = SumoEnvironment(
            net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
            route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
            out_csv_name=f"outputs/2way-single-intersection/dqn_seed_{seed}",
            single_agent=True,
            use_gui=False,        # GUI off for speed
            num_seconds=100000,
        )

        model = DQN(
            env=env,
            policy="MlpPolicy",
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            verbose=1,
            seed=seed,            # seed set here
        )

        model.learn(total_timesteps=100000)
        model.save(f"outputs/2way-single-intersection/dqn_model_seed_{seed}")

        env.close()
        print(f"Seed {seed} complete. CSV and model saved.")

    print("\n===== ALL 5 BASELINE RUNS COMPLETE =====")
    print("Check outputs/2way-single-intersection/ for results")