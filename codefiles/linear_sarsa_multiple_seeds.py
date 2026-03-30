# linear_sarsa_multiple_seeds.py
# semi-gradient SARSA with tile coding on the sumo-rl intersection.
# on-policy: updates using the actual next action taken, not the best one.
# tends to learn safer policies since it accounts for exploration mistakes.
# runs across same 5 seeds as the DQN baseline for fair comparison.

import os
import sys
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


class TileCoder:
    def __init__(self, obs_low, obs_high, num_tilings=8, num_tiles=4, seed=None):
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.obs_low = np.array(obs_low, dtype=np.float32)
        self.obs_high = np.array(obs_high, dtype=np.float32)
        self.hash_size = 4096
        self.total_size = self.num_tilings * self.hash_size
        rng = np.random.RandomState(seed)
        self.offsets = rng.uniform(0, 1, size=(num_tilings, len(obs_low)))

    def get_features(self, obs):
        obs = np.clip(np.array(obs, dtype=np.float32), self.obs_low, self.obs_high)
        rng = self.obs_high - self.obs_low
        rng[rng == 0] = 1.0
        normalized = (obs - self.obs_low) / rng

        active_tiles = []
        for tiling in range(self.num_tilings):
            shifted = (normalized + self.offsets[tiling]) * self.num_tiles
            coords = tuple(int(s) for s in shifted)
            idx = tiling * self.hash_size + (hash(coords) % self.hash_size)
            active_tiles.append(idx)
        return active_tiles


class LinearSARSAAgent:
    # Q(s, a) = sum of weights at active tile indices
    # target = reward + gamma * Q(s', a')  <-- actual next action, not max

    def __init__(self, num_actions, tile_coder, alpha=0.05, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.9999, epsilon_min=0.01, seed=None):
        self.num_actions = num_actions
        self.tc = tile_coder
        self.alpha = alpha / tile_coder.num_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weights = np.zeros((num_actions, tile_coder.total_size))
        self.rng = np.random.RandomState(seed)

    def q_value(self, obs, action):
        return sum(self.weights[action][f] for f in self.tc.get_features(obs))

    def get_action(self, obs):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.num_actions)
        q_vals = [self.q_value(obs, a) for a in range(self.num_actions)]
        return int(np.argmax(q_vals))

    def update(self, obs, action, reward, next_obs, next_action, done):
        features = self.tc.get_features(obs)
        q_current = sum(self.weights[action][f] for f in features)

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_value(next_obs, next_action)

        td_error = target - q_current
        for f in features:
            self.weights[action][f] += self.alpha * td_error

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    os.makedirs("outputs/linear-sarsa-2way-single-intersection", exist_ok=True)

    seeds = [42, 123, 456, 789, 1000]

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Running Linear SARSA with seed: {seed}")
        print(f"{'='*40}")

        env = SumoEnvironment(
            net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
            route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
            out_csv_name=f"outputs/linear-sarsa-2way-single-intersection/linear_sarsa_seed_{seed}",
            single_agent=True,
            use_gui=False,
            num_seconds=100000,
        )

        obs_dim = env.observation_space.shape[0]
        num_actions = env.action_space.n
        tc = TileCoder(np.zeros(obs_dim), np.ones(obs_dim), seed=seed)
        agent = LinearSARSAAgent(num_actions, tc, seed=seed)

        for ep in range(5):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            action = agent.get_action(obs)

            while not done:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = agent.get_action(next_obs)
                agent.update(obs, action, reward, next_obs, next_action, done)
                obs = next_obs
                action = next_action
                total_reward += reward
                steps += 1

            print(f"  ep {ep+1}/5: reward={total_reward:.2f}, steps={steps}, eps={agent.epsilon:.4f}")

        env.close()
        print(f"Seed {seed} complete.")

    print("\n===== ALL 5 LINEAR SARSA RUNS COMPLETE =====")
