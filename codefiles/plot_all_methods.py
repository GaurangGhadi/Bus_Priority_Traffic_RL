# plot_all_methods.py
# generates comparison plots across DQN, Linear Q, and Linear SARSA
# for both baseline and bus priority experiments.

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots_all_methods", exist_ok=True)

configs = [
    ("DQN Baseline", "dqn-2way-single-intersection", "dqn_seed_{seed}_conn*"),
    ("DQN β=0.1", "dqn-2way-single-intersection", "dqn_bus_b01_seed_{seed}_conn*"),
    ("DQN β=1.0", "dqn-2way-single-intersection", "dqn_bus_b10_seed_{seed}_conn*"),
    ("DQN β=2.0", "dqn-2way-single-intersection", "dqn_bus_b20_seed_{seed}_conn*"),
    ("LinQ Baseline", "linear-q-2way-single-intersection", "linear_q_seed_{seed}_conn*"),
    ("LinQ β=0.1", "linear-q-2way-single-intersection", "linear_q_bus_b01_seed_{seed}_conn*"),
    ("LinQ β=1.0", "linear-q-2way-single-intersection", "linear_q_bus_b10_seed_{seed}_conn*"),
    ("LinQ β=2.0", "linear-q-2way-single-intersection", "linear_q_bus_b20_seed_{seed}_conn*"),
    ("SARSA Baseline", "linear-sarsa-2way-single-intersection", "linear_sarsa_seed_{seed}_conn*"),
    ("SARSA β=0.1", "linear-sarsa-2way-single-intersection", "linear_sarsa_bus_b01_seed_{seed}_conn*"),
    ("SARSA β=1.0", "linear-sarsa-2way-single-intersection", "linear_sarsa_bus_b10_seed_{seed}_conn*"),
    ("SARSA β=2.0", "linear-sarsa-2way-single-intersection", "linear_sarsa_bus_b20_seed_{seed}_conn*"),
]
seeds = [42, 123, 456, 789, 1000]

# load all data
data = {}
for name, folder, pattern in configs:
    seed_waits = []
    seed_arrived = []
    seed_ep_waits = []
    seed_ep_arrived = []
    for seed in seeds:
        p = pattern.format(seed=seed)
        files = sorted(glob.glob(f"outputs/{folder}/{p}_ep*.csv"))
        if not files:
            continue
        # last episode
        f = files[-1]
        waits = []
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
            for row in rows:
                waits.append(float(row["system_mean_waiting_time"]))
            last = rows[-1]
        seed_waits.append(np.mean(waits))
        seed_arrived.append(int(last["system_total_arrived"]))

        # all episodes for learning curve
        ep_w, ep_a = [], []
        for ef in files:
            with open(ef) as fh2:
                rows2 = list(csv.DictReader(fh2))
                ep_w.append(np.mean([float(r["system_mean_waiting_time"]) for r in rows2]))
                ep_a.append(int(rows2[-1]["system_total_arrived"]))
        seed_ep_waits.append(ep_w)
        seed_ep_arrived.append(ep_a)

    data[name] = {
        "waits": seed_waits,
        "arrived": seed_arrived,
        "ep_waits": seed_ep_waits,
        "ep_arrived": seed_ep_arrived,
    }

print("loaded all data")


# --- plot 1: baseline comparison (all 3 methods) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
baselines = ["DQN Baseline", "LinQ Baseline", "SARSA Baseline"]
bl_labels = ["DQN", "Linear Q", "Linear SARSA"]
bl_colors = ["#2196F3", "#4CAF50", "#FF9800"]

wait_means = [np.mean(data[b]["waits"]) for b in baselines]
wait_stds = [np.std(data[b]["waits"]) for b in baselines]
arr_means = [np.mean(data[b]["arrived"]) for b in baselines]
arr_stds = [np.std(data[b]["arrived"]) for b in baselines]

axes[0].bar(bl_labels, wait_means, yerr=wait_stds, capsize=5, color=bl_colors)
axes[0].set_ylabel("Avg System Mean Waiting Time")
axes[0].set_title("Baseline Wait (No Buses)")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(bl_labels, arr_means, yerr=arr_stds, capsize=5, color=bl_colors)
axes[1].set_ylabel("Total Vehicles Arrived")
axes[1].set_title("Baseline Throughput (No Buses)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots_all_methods/1_baseline_comparison.png", dpi=150)
plt.close()
print("saved 1_baseline_comparison.png")


# --- plot 2: bus priority comparison across methods (grouped by beta) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
methods = ["DQN", "LinQ", "SARSA"]
betas_list = ["β=0.1", "β=1.0", "β=2.0"]
method_colors = ["#2196F3", "#4CAF50", "#FF9800"]

x = np.arange(len(betas_list))
width = 0.25

for i, method in enumerate(methods):
    w_means, w_stds, a_means, a_stds = [], [], [], []
    for beta in betas_list:
        key = f"{method} {beta}"
        w_means.append(np.mean(data[key]["waits"]))
        w_stds.append(np.std(data[key]["waits"]))
        a_means.append(np.mean(data[key]["arrived"]))
        a_stds.append(np.std(data[key]["arrived"]))

    axes[0].bar(x + i * width, w_means, width, yerr=w_stds, capsize=4,
                label=method, color=method_colors[i])
    axes[1].bar(x + i * width, a_means, width, yerr=a_stds, capsize=4,
                label=method, color=method_colors[i])

axes[0].set_xlabel("β (Bus Priority Weight)")
axes[0].set_ylabel("Avg System Mean Waiting Time")
axes[0].set_title("Bus Priority: Wait Time by Method and β")
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(betas_list)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

axes[1].set_xlabel("β (Bus Priority Weight)")
axes[1].set_ylabel("Total Vehicles Arrived")
axes[1].set_title("Bus Priority: Throughput by Method and β")
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(betas_list)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots_all_methods/2_bus_priority_by_method.png", dpi=150)
plt.close()
print("saved 2_bus_priority_by_method.png")


# --- plot 3: variance comparison (std across seeds) ---
fig, ax = plt.subplots(figsize=(12, 5))

all_configs_short = ["BL", "β=0.1", "β=1.0", "β=2.0"]
x = np.arange(len(all_configs_short))
width = 0.25

for i, method in enumerate(methods):
    stds = []
    keys = [f"{method} Baseline", f"{method} β=0.1", f"{method} β=1.0", f"{method} β=2.0"]
    for key in keys:
        stds.append(np.std(data[key]["waits"]))
    ax.bar(x + i * width, stds, width, label=method, color=method_colors[i])

ax.set_xlabel("Configuration")
ax.set_ylabel("Std of Wait Time Across Seeds")
ax.set_title("Variance Comparison — Lower = More Consistent")
ax.set_xticks(x + width)
ax.set_xticklabels(all_configs_short)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots_all_methods/3_variance_comparison.png", dpi=150)
plt.close()
print("saved 3_variance_comparison.png")


# --- plot 4: beta sensitivity per method ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
betas_num = [0.0, 0.1, 1.0, 2.0]

for i, method in enumerate(methods):
    keys = [f"{method} Baseline", f"{method} β=0.1", f"{method} β=1.0", f"{method} β=2.0"]
    w_means = [np.mean(data[k]["waits"]) for k in keys]
    w_stds = [np.std(data[k]["waits"]) for k in keys]
    a_means = [np.mean(data[k]["arrived"]) for k in keys]
    a_stds = [np.std(data[k]["arrived"]) for k in keys]

    ax = axes[i]
    ax2 = ax.twinx()
    l1 = ax.errorbar(betas_num, w_means, yerr=w_stds, marker="o", capsize=5,
                      color="#F44336", linewidth=2, markersize=8, label="Wait")
    l2 = ax2.errorbar(betas_num, a_means, yerr=a_stds, marker="s", capsize=5,
                       color="#2196F3", linewidth=2, markersize=8, label="Arrived")
    ax.set_xlabel("β")
    ax.set_ylabel("Wait Time", color="#F44336")
    ax2.set_ylabel("Vehicles Arrived", color="#2196F3")
    ax.set_title(f"{method}")
    ax.grid(alpha=0.3)
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left")

plt.tight_layout()
plt.savefig("plots_all_methods/4_beta_sensitivity_per_method.png", dpi=150)
plt.close()
print("saved 4_beta_sensitivity_per_method.png")


# --- plot 5: learning curves across episodes (baseline, mean over seeds) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, bl in enumerate(baselines):
    ep_waits = data[bl]["ep_waits"]
    ep_arrived = data[bl]["ep_arrived"]
    if not ep_waits:
        continue
    max_eps = max(len(e) for e in ep_waits)
    mean_w, std_w, mean_a, std_a = [], [], [], []
    for ep_idx in range(max_eps):
        w = [ep_waits[s][ep_idx] for s in range(len(ep_waits)) if ep_idx < len(ep_waits[s])]
        a = [ep_arrived[s][ep_idx] for s in range(len(ep_arrived)) if ep_idx < len(ep_arrived[s])]
        mean_w.append(np.mean(w))
        std_w.append(np.std(w))
        mean_a.append(np.mean(a))
        std_a.append(np.std(a))

    eps_x = np.arange(1, max_eps + 1)
    mean_w, std_w = np.array(mean_w), np.array(std_w)
    mean_a, std_a = np.array(mean_a), np.array(std_a)

    axes[0].plot(eps_x, mean_w, marker="o", color=bl_colors[i], label=bl_labels[i])
    axes[0].fill_between(eps_x, mean_w - std_w, mean_w + std_w, color=bl_colors[i], alpha=0.15)
    axes[1].plot(eps_x, mean_a, marker="o", color=bl_colors[i], label=bl_labels[i])
    axes[1].fill_between(eps_x, mean_a - std_a, mean_a + std_a, color=bl_colors[i], alpha=0.15)

axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Avg Wait Time")
axes[0].set_title("Baseline Learning Curve (Mean ± Std)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Vehicles Arrived")
axes[1].set_title("Baseline Throughput Curve (Mean ± Std)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots_all_methods/5_baseline_learning_curves.png", dpi=150)
plt.close()
print("saved 5_baseline_learning_curves.png")


# --- plot 6: best bus priority config per method ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# DQN best = b2.0, Linear Q best = b0.1 (lowest wait), SARSA best = b0.1
best_configs = ["DQN β=2.0", "LinQ β=0.1", "SARSA β=0.1"]
best_labels = ["DQN (β=2.0)", "Linear Q (β=0.1)", "SARSA (β=0.1)"]

w_means = [np.mean(data[b]["waits"]) for b in best_configs]
w_stds = [np.std(data[b]["waits"]) for b in best_configs]
a_means = [np.mean(data[b]["arrived"]) for b in best_configs]
a_stds = [np.std(data[b]["arrived"]) for b in best_configs]

axes[0].bar(best_labels, w_means, yerr=w_stds, capsize=5, color=bl_colors)
axes[0].set_ylabel("Avg System Mean Waiting Time")
axes[0].set_title("Best Bus Priority Config per Method — Wait")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(best_labels, a_means, yerr=a_stds, capsize=5, color=bl_colors)
axes[1].set_ylabel("Total Vehicles Arrived")
axes[1].set_title("Best Bus Priority Config per Method — Throughput")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots_all_methods/6_best_bus_priority_per_method.png", dpi=150)
plt.close()
print("saved 6_best_bus_priority_per_method.png")


# --- plot 7: boxplots per method (bus priority b=2.0 since DQN best there) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bp_configs = ["DQN β=2.0", "LinQ β=2.0", "SARSA β=2.0"]
bp_labels = ["DQN", "Linear Q", "SARSA"]

wait_box = [data[c]["waits"] for c in bp_configs]
arr_box = [data[c]["arrived"] for c in bp_configs]

bp1 = axes[0].boxplot(wait_box, labels=bp_labels, patch_artist=True)
for patch, color in zip(bp1["boxes"], bl_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_ylabel("Avg Wait Time")
axes[0].set_title("Wait Distribution at β=2.0 (5 Seeds)")
axes[0].grid(axis="y", alpha=0.3)

bp2 = axes[1].boxplot(arr_box, labels=bp_labels, patch_artist=True)
for patch, color in zip(bp2["boxes"], bl_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel("Vehicles Arrived")
axes[1].set_title("Throughput Distribution at β=2.0 (5 Seeds)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots_all_methods/7_boxplots_b20.png", dpi=150)
plt.close()
print("saved 7_boxplots_b20.png")

print("\ndone! all plots in plots_all_methods/")
