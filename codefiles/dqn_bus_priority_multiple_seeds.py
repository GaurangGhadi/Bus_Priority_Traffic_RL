# dqn_bus_priority_multiple_seeds.py
# runs bus priority DQN across multiple seeds and beta values.
# same hyperparams as tapan's baseline script, just with the wrapper added.

import os
import sys

from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

sys.path.insert(0, os.path.expanduser("~/Reinforcement Learning/Project"))
from bus_priority import BusPriorityWrapper, generate_bus_routes


if __name__ == "__main__":

    net_file = "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml"
    route_file = "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml"
    bus_route_file = route_file.replace(".rou.xml", "-with-buses.rou.xml")

    if not os.path.exists(bus_route_file):
        generate_bus_routes(route_file, bus_route_file,
                           net_file=net_file, bus_frequency=300,
                           simulation_end=100000)

    seeds = [42, 123, 456, 789, 1000]
    betas = [0.1, 1.0, 2.0]

    for beta in betas:
        # label for filenames: 0.1 -> "b01", 1.0 -> "b1", 2.0 -> "b2"
        beta_label = f"b{str(beta).replace('.', '')}"

        print(f"\n{'='*50}")
        print(f"Starting beta={beta} runs ({beta_label})")
        print(f"{'='*50}")

        for seed in seeds:
            print(f"\n--- beta={beta}, seed={seed} ---")

            env = SumoEnvironment(
                net_file=net_file,
                route_file=bus_route_file,
                out_csv_name=f"outputs/2way-single-intersection/dqn_bus_{beta_label}_seed_{seed}",
                single_agent=True,
                use_gui=False,
                num_seconds=100000,
            )

            env = BusPriorityWrapper(env, beta=beta)

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
                seed=seed,
            )

            model.learn(total_timesteps=100000)
            model.save(f"outputs/2way-single-intersection/dqn_bus_{beta_label}_model_seed_{seed}")

            env.close()
            print(f"beta={beta}, seed={seed} complete.")

        print(f"\n===== ALL 5 SEEDS COMPLETE FOR beta={beta} =====")

    print("\n===== ALL RUNS COMPLETE =====")
    print("Files saved as dqn_bus_b01_seed_*, dqn_bus_b10_seed_*, dqn_bus_b20_seed_*")
