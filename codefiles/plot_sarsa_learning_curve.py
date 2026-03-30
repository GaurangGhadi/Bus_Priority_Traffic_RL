# plot_sarsa_learning_curve.py
# generates a learning curve for Linear SARSA matching the style of
# Tapan's DQN v2 and DDQN learning curve plots (bold=mean, shaded=std, faded=individual seeds)

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

seeds = [42, 123, 456, 789, 1000]
folder = "outputs/linear-sarsa-2way-single-intersection"
pattern = "linear_sarsa_seed_{seed}_conn*"

# collect per-episode system total waiting time for each seed
seed_episodes = {}
for seed in seeds:
    p = pattern.format(seed=seed)
    files = sorted(glob.glob(f"{folder}/{p}_ep*.csv"))
    ep_totals = []
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
            last = rows[-1]
            ep_totals.append(float(last["system_total_waiting_time"]))
    seed_episodes[seed] = ep_totals

# compute mean and std across seeds per episode
max_eps = max(len(v) for v in seed_episodes.values())
all_ep = np.array([seed_episodes[s] for s in seeds])  # shape: (5, num_eps)

mean_vals = np.mean(all_ep, axis=0)
std_vals = np.std(all_ep, axis=0)
eps_x = np.arange(1, max_eps + 1)

# plot
fig, ax = plt.subplots(figsize=(8, 5))

# individual seeds (faded)
for seed in seeds:
    ax.plot(eps_x, seed_episodes[seed], color="tab:green", alpha=0.2, linewidth=1)

# mean (bold) + std (shaded)
ax.plot(eps_x, mean_vals, color="tab:green", linewidth=2.5, label="Linear SARSA")
ax.fill_between(eps_x, mean_vals - std_vals, mean_vals + std_vals,
                color="tab:green", alpha=0.25)

ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("System Total Waiting Time (s)", fontsize=12)
ax.set_title("Linear SARSA — Learning curve with spread\n(bold = mean, shaded = std, faded = individual seeds)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(eps_x)

plt.tight_layout()
plt.savefig("plots_all_methods/linear_sarsa_learning_curve_spread.png", dpi=150)
plt.close()
print("saved plots_all_methods/linear_sarsa_learning_curve_spread.png")
