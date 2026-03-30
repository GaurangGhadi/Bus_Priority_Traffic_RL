# run_mcts.py
# Monte Carlo Tree Search for traffic signal control.
# Uses SUMO as a world model — saves state, simulates rollouts,
# restores state, then executes the best action.
# Depth 3: evaluates each of 4 actions 3 steps ahead.

import os
import sys
import random
import numpy as np
import pandas as pd

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

os.chdir("/home/tapan/classes/CS5180/Final_Project/sumo-rl")


class MCTSAgent:
    """
    Pure MCTS agent using SUMO as world model.
    At each step:
      1. Save SUMO state
      2. For each action, simulate ROLLOUT_DEPTH steps and sum rewards
      3. Restore SUMO state
      4. Return action with highest cumulative reward
    """

    def __init__(self, action_dim, rollout_depth=3, state_file="/tmp/mcts_sumo_state.xml"):
        self.action_dim    = action_dim
        self.rollout_depth = rollout_depth
        self.state_file    = state_file

    def select_action(self, env):
        """evaluate all actions via shallow rollout, return best."""
        ts_id = list(env.traffic_signals.keys())[0]
        sumo  = env.traffic_signals[ts_id].sumo

        # save current state
        sumo.simulation.saveState(self.state_file)

        best_action = 0
        best_reward = -np.inf

        for action in range(self.action_dim):
            # restore to saved state before each rollout
            sumo.simulation.loadState(self.state_file)

            cumulative_reward = 0.0
            current_action    = action

            for depth in range(self.rollout_depth):
                _, reward, terminated, truncated, _ = env.step(current_action)
                cumulative_reward += reward

                if terminated or truncated:
                    break

                # random rollout policy after first action
                current_action = random.randint(0, self.action_dim - 1)

            if cumulative_reward > best_reward:
                best_reward = cumulative_reward
                best_action = action

        # restore state one final time before executing real action
        sumo.simulation.loadState(self.state_file)

        return best_action


def run_mcts_seed(seed, net_file, route_file, output_dir,
                  rollout_depth=3, total_seconds=100000):
    print(f"\n{'='*40}")
    print(f"MCTS — Seed: {seed} | Depth: {rollout_depth}")
    print(f"{'='*40}")

    random.seed(seed)
    np.random.seed(seed)

    env = SumoEnvironment(
        net_file     = net_file,
        route_file   = route_file,
        out_csv_name = f"{output_dir}/mcts_seed_{seed}",
        single_agent = True,
        use_gui      = False,
        num_seconds  = total_seconds,
    )

    action_dim = env.action_space.n
    agent      = MCTSAgent(action_dim=action_dim, rollout_depth=rollout_depth)

    print(f"Action dim: {action_dim} | Rollout depth: {rollout_depth}")

    state, _          = env.reset()
    episode           = 0
    step              = 0
    episode_waittimes = []

    while True:
        # MCTS selects best action via rollouts
        action = agent.select_action(env)

        # execute real action
        next_state, reward, terminated, truncated, info = env.step(action)
        done   = terminated or truncated
        step  += 1
        state  = next_state

        if done:
            episode += 1
            avg_wait = info.get("system_total_waiting_time", 0)
            episode_waittimes.append(avg_wait)
            print(f"  Episode {episode:3d} | "
                  f"Steps: {step:6d} | "
                  f"Waiting: {avg_wait:.1f}s")
            state, _ = env.reset()
            if step >= total_seconds:
                break

    env.close()
    print(f"Seed {seed} complete.")
    return episode_waittimes


if __name__ == "__main__":
    net_file   = "sumo_rl/nets/2way-single-intersection/single-intersection.net.xml"
    route_file = "sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml"
    output_dir = "project/output/mcts"
    os.makedirs(output_dir, exist_ok=True)

    SEEDS         = [42, 123, 456, 789, 1000]
    ROLLOUT_DEPTH = 1       # depth 3 was too slow (~5.5hrs/episode)
    TOTAL_SECONDS = 10000   # reduced from 100000 — saveState/loadState
                            # latency scales with simulation size:
                            # 73ms at step 900 → 409ms at step 13500
                            # making 100k steps infeasible
    all_results   = {}

    for seed in SEEDS:
        all_results[seed] = run_mcts_seed(
            seed          = seed,
            net_file      = net_file,
            route_file    = route_file,
            output_dir    = output_dir,
            rollout_depth = ROLLOUT_DEPTH,
            total_seconds = TOTAL_SECONDS,
        )

    print("\n===== ALL MCTS RUNS COMPLETE =====")
    print(f"\n{'Seed':<6} {'Episodes':>10} {'Final wait':>20} {'Mean wait':>20}")
    print("-"*60)
    final_waits = []
    mean_waits  = []
    for seed in SEEDS:
        eps   = all_results[seed]
        final = eps[-1] if eps else 0
        mean  = np.mean(eps) if eps else 0
        final_waits.append(final)
        mean_waits.append(mean)
        print(f"{seed:<6} {len(eps):>10} {final:>20.1f}s {mean:>20.1f}s")

    print(f"\n{'':6} {'':>10} {'':>20} {'':>20}")
    print(f"{'Mean':<6} {'':>10} {np.mean(final_waits):>20.1f}s {np.mean(mean_waits):>20.1f}s")
    print(f"{'Std':<6} {'':>10} {np.std(final_waits):>20.1f}s {np.std(mean_waits):>20.1f}s")

    # save per-seed episode results
    rows = []
    for seed in SEEDS:
        for ep, wait in enumerate(all_results[seed], 1):
            rows.append({"seed": seed, "episode": ep, "waiting_time": wait})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{output_dir}/mcts_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/mcts_results.csv")

    # save summary
    summary = {
        "algorithm":    "MCTS",
        "rollout_depth": ROLLOUT_DEPTH,
        "total_seconds": TOTAL_SECONDS,
        "note":         "saveState/loadState latency made depth>1 and num_seconds>10000 infeasible",
        "seeds":        SEEDS,  
        "final_waits":  final_waits,
        "mean":         float(np.mean(final_waits)),
        "std":          float(np.std(final_waits)),
    }
    import json
    with open(f"{output_dir}/mcts_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir}/mcts_summary.json")