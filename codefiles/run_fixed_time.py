import os
import sys
import numpy as np
import pandas as pd

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

os.chdir("/home/tapan/classes/CS5180/Final_Project/sumo-rl")

if __name__ == "__main__":
    seeds   = [42, 123, 456, 789, 1000]
    results = {}

    net_file   = "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml"
    route_file = "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml"
    output_dir = "project/output/fixed_time"
    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Fixed-time control — Seed: {seed}")
        print(f"{'='*40}")

        import random
        random.seed(seed)
        np.random.seed(seed)

        env = SumoEnvironment(
            net_file     = net_file,
            route_file   = route_file,
            out_csv_name = f"{output_dir}/fixed_time_seed_{seed}",
            single_agent = True,
            use_gui      = False,
            num_seconds  = 100000,
            # fixed green time per phase — no min green enforcement
            min_green    = 30,
            max_green    = 30,
        )

        state, _      = env.reset()
        action_dim    = env.action_space.n
        done          = False
        current_phase = 0
        step          = 0
        episode_waits = []

        while True:
            # fixed policy: cycle through phases in order
            action = current_phase % action_dim

            next_state, reward, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            step += 1

            if done:
                wait = info.get("system_total_waiting_time", 0)
                episode_waits.append(wait)
                print(f"  Episode {len(episode_waits):3d} | "
                      f"Steps: {step:6d} | "
                      f"Waiting: {wait:.1f}s")
                current_phase += 1
                state, _ = env.reset()
                if step >= 100000:
                    break
            else:
                state = next_state

        results[seed] = episode_waits
        env.close()
        print(f"Seed {seed} complete.")

    print("\n===== FIXED TIME COMPLETE =====")
    print(f"\n{'Seed':<6} {'Final episode wait':>20}")
    print("-"*30)
    final_waits = []
    for seed in seeds:
        final = results[seed][-1] if results[seed] else 0
        final_waits.append(final)
        print(f"{seed:<6} {final:>20.1f}s")

    print(f"\nMean: {np.mean(final_waits):.1f}s")
    print(f"Std:  {np.std(final_waits):.1f}s")