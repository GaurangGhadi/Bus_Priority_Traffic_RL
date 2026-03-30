# plot_multiseed.py
# generates comparison plots from the multi-seed baseline + bus priority runs.

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots_multiseed", exist_ok=True)

configs = [
    ("Baseline\n(no buses)", "dqn_seed_{seed}_conn*"),
    ("Bus β=0.1", "dqn_bus_b01_seed_{seed}_conn*"),
    ("Bus β=1.0", "dqn_bus_b10_seed_{seed}_conn*"),
    ("Bus β=2.0", "dqn_bus_b20_seed_{seed}_conn*"),
]
seeds = [42, 123, 456, 789, 1000]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
config_names_short = ["Baseline", "β=0.1", "β=1.0", "β=2.0"]

# load all data
data = {}
for name, pattern in configs:
    seed_data = {}
    for seed in seeds:
        p = pattern.format(seed=seed)
        files = sorted(glob.glob(f"outputs/2way-single-intersection/{p}_ep*.csv"))
        if not files:
            continue
        episodes = []
        for f in files:
            waits = []
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
                for row in rows:
                    waits.append(float(row["system_mean_waiting_time"]))
                last = rows[-1]
            episodes.append({
                "avg_wait": np.mean(waits),
                "arrived": int(last["system_total_arrived"]),
                "final_wait": float(last["system_mean_waiting_time"]),
                "waits_over_time": waits,
            })
        seed_data[seed] = episodes
    data[name] = seed_data

print("loaded data:")
for name in data:
    print(f"  {name.replace(chr(10), ' ')}: {len(data[name])} seeds")


# --- plot 1: avg wait per seed (last episode), grouped bar ---
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(seeds))
width = 0.2

for i, (name, _) in enumerate(configs):
    vals = []
    for seed in seeds:
        eps = data[name].get(seed, [])
        if eps:
            vals.append(eps[-1]["avg_wait"])  # last episode
        else:
            vals.append(0)
    ax.bar(x + i * width, vals, width, label=name.replace("\n", " "), color=colors[i])

ax.set_xlabel("Seed")
ax.set_ylabel("Avg System Mean Waiting Time")
ax.set_title("Average Wait Time per Seed (Last Episode)")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([str(s) for s in seeds])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_multiseed/1_wait_per_seed.png", dpi=150)
plt.close()
print("saved 1_wait_per_seed.png")


# --- plot 2: throughput per seed (last episode) ---
fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, _) in enumerate(configs):
    vals = []
    for seed in seeds:
        eps = data[name].get(seed, [])
        if eps:
            vals.append(eps[-1]["arrived"])
        else:
            vals.append(0)
    ax.bar(x + i * width, vals, width, label=name.replace("\n", " "), color=colors[i])

ax.set_xlabel("Seed")
ax.set_ylabel("Total Vehicles Arrived")
ax.set_title("Throughput per Seed (Last Episode)")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([str(s) for s in seeds])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_multiseed/2_throughput_per_seed.png", dpi=150)
plt.close()
print("saved 2_throughput_per_seed.png")


# --- plot 3: mean +/- std summary (last episode across seeds) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

wait_means, wait_stds = [], []
arr_means, arr_stds = [], []

for name, _ in configs:
    waits = [data[name][s][-1]["avg_wait"] for s in seeds if s in data[name]]
    arrived = [data[name][s][-1]["arrived"] for s in seeds if s in data[name]]
    wait_means.append(np.mean(waits))
    wait_stds.append(np.std(waits))
    arr_means.append(np.mean(arrived))
    arr_stds.append(np.std(arrived))

x_pos = np.arange(len(config_names_short))
axes[0].bar(x_pos, wait_means, yerr=wait_stds, capsize=5, color=colors)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(config_names_short)
axes[0].set_ylabel("Avg System Mean Waiting Time")
axes[0].set_title("Mean Wait ± Std (5 Seeds, Last Episode)")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x_pos, arr_means, yerr=arr_stds, capsize=5, color=colors)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(config_names_short)
axes[1].set_ylabel("Total Vehicles Arrived")
axes[1].set_title("Mean Throughput ± Std (5 Seeds, Last Episode)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots_multiseed/3_summary_mean_std.png", dpi=150)
plt.close()
print("saved 3_summary_mean_std.png")


# --- plot 4: beta sensitivity with error bars ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
betas = [0.0, 0.1, 1.0, 2.0]

axes[0].errorbar(betas, wait_means, yerr=wait_stds, marker="o", capsize=5,
                 color="#F44336", linewidth=2, markersize=8)
axes[0].set_xlabel("β (Bus Priority Weight)")
axes[0].set_ylabel("Avg System Mean Waiting Time")
axes[0].set_title("Wait Time vs β (5 Seeds)")
axes[0].grid(alpha=0.3)

axes[1].errorbar(betas, arr_means, yerr=arr_stds, marker="o", capsize=5,
                 color="#2196F3", linewidth=2, markersize=8)
axes[1].set_xlabel("β (Bus Priority Weight)")
axes[1].set_ylabel("Total Vehicles Arrived")
axes[1].set_title("Throughput vs β (5 Seeds)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots_multiseed/4_beta_sensitivity.png", dpi=150)
plt.close()
print("saved 4_beta_sensitivity.png")


# --- plot 5: learning curves across episodes (averaged over seeds) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (name, _) in enumerate(configs):
    ep_waits = []
    ep_arrived = []
    for seed in seeds:
        if seed not in data[name]:
            continue
        eps = data[name][seed]
        ep_waits.append([ep["avg_wait"] for ep in eps])
        ep_arrived.append([ep["arrived"] for ep in eps])

    if not ep_waits:
        continue
    # pad to same length if needed
    max_eps = max(len(e) for e in ep_waits)
    mean_waits = []
    std_waits = []
    mean_arr = []
    std_arr = []
    for ep_idx in range(max_eps):
        w_vals = [ep_waits[s][ep_idx] for s in range(len(ep_waits)) if ep_idx < len(ep_waits[s])]
        a_vals = [ep_arrived[s][ep_idx] for s in range(len(ep_arrived)) if ep_idx < len(ep_arrived[s])]
        mean_waits.append(np.mean(w_vals))
        std_waits.append(np.std(w_vals))
        mean_arr.append(np.mean(a_vals))
        std_arr.append(np.std(a_vals))

    eps_x = np.arange(1, max_eps + 1)
    mean_waits = np.array(mean_waits)
    std_waits = np.array(std_waits)
    mean_arr = np.array(mean_arr)
    std_arr = np.array(std_arr)

    label = name.replace("\n", " ")
    axes[0].plot(eps_x, mean_waits, marker="o", color=colors[i], label=label)
    axes[0].fill_between(eps_x, mean_waits - std_waits, mean_waits + std_waits,
                         color=colors[i], alpha=0.15)
    axes[1].plot(eps_x, mean_arr, marker="o", color=colors[i], label=label)
    axes[1].fill_between(eps_x, mean_arr - std_arr, mean_arr + std_arr,
                         color=colors[i], alpha=0.15)

axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Avg Wait Time")
axes[0].set_title("Wait Time Across Episodes (Mean ± Std over Seeds)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Vehicles Arrived")
axes[1].set_title("Throughput Across Episodes (Mean ± Std over Seeds)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots_multiseed/5_learning_across_episodes.png", dpi=150)
plt.close()
print("saved 5_learning_across_episodes.png")


# --- plot 6: box plots across seeds ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

wait_data_box = []
arr_data_box = []
for name, _ in configs:
    waits = [data[name][s][-1]["avg_wait"] for s in seeds if s in data[name]]
    arrived = [data[name][s][-1]["arrived"] for s in seeds if s in data[name]]
    wait_data_box.append(waits)
    arr_data_box.append(arrived)

bp1 = axes[0].boxplot(wait_data_box, labels=config_names_short, patch_artist=True)
for patch, color in zip(bp1["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_ylabel("Avg System Mean Waiting Time")
axes[0].set_title("Wait Time Distribution (5 Seeds, Last Episode)")
axes[0].grid(axis="y", alpha=0.3)

bp2 = axes[1].boxplot(arr_data_box, labels=config_names_short, patch_artist=True)
for patch, color in zip(bp2["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel("Total Vehicles Arrived")
axes[1].set_title("Throughput Distribution (5 Seeds, Last Episode)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots_multiseed/6_boxplots.png", dpi=150)
plt.close()
print("saved 6_boxplots.png")

print("\ndone! all plots in plots_multiseed/")
