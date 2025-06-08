import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a DataFrame from the provided data in the image

data = {
    "Size of action space": [11, 21, 31, 41, 61, 81, 101],
    "reward1": [0.838, 0.948, 0.96, 0.9, 0.98, 0.94, 0.962],
    "reward2": [0.822, 1.026, 0.984, 1.044, 0.864, 1.069, 1.002],
    "reward3": [0.758, 1.014, 0.968, 0.9, 0.888, 0.992, 1.1],
    "Convergence1": [7.02, 4.8, 4.6, 3.64, 4.18, 2.42, 2.82],
    "Convergence2": [6.76, 6.44, 4.66, 4.84, 3.2, 2.06, 2.62],
    "Convergence3": [6.22, 4.64, 5.32, 5.2, 3.64, 1.84, 2.56]
}

df = pd.DataFrame(data)

# Compute the mean and standard deviation for rewards and convergence
df['reward_mean'] = df[['reward1', 'reward2', 'reward3']].mean(axis=1)
df['reward_std'] = df[['reward1', 'reward2', 'reward3']].std(axis=1)

df['convergence_mean'] = df[['Convergence1', 'Convergence2', 'Convergence3']].mean(axis=1)
df['convergence_std'] = df[['Convergence1', 'Convergence2', 'Convergence3']].std(axis=1)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

font_size = 20

# Plot action space vs reward
ax1.set_xlabel('Action Space Size', fontsize=font_size)
ax1.set_ylabel('Reward', fontsize=font_size)
ax1.plot(df['Size of action space'], df['reward_mean'], 'o-', color='tab:blue', label='Mean Reward')
ax1.fill_between(df['Size of action space'], df['reward_mean'] - df['reward_std'], df['reward_mean'] + df['reward_std'], color='tab:blue', alpha=0.3)
# ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=font_size)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Create a second y-axis for convergence
ax2.set_xlabel('Action Space Size', fontsize=font_size)
ax2.set_ylabel('Convergence Steps', fontsize=font_size)
ax2.plot(df['Size of action space'], df['convergence_mean'], 's-', color='tab:red', label='Mean Convergence')
ax2.fill_between(df['Size of action space'], df['convergence_mean'] - df['convergence_std'], df['convergence_mean'] + df['convergence_std'], color='tab:red', alpha=0.3)
# ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=font_size)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)


fig.tight_layout()
plt.savefig('./action_space_plots.pdf')
