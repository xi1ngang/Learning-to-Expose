import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Statistical Reward': [0.8, 0.9, 0.85, 0.87, 0.88, 0.92, 0.89, 0.91, 0.86, 0.84],
    'Feature Point Count Reward': [0.75, 0.82, 0.8, 0.83, 0.81, 0.85, 0.84, 0.86, 0.79, 0.77],
    'Object Detection Reward': [0.78, 0.85, 0.82, 0.86, 0.84, 0.88, 0.87, 0.89, 0.83, 0.81]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the first scatter plot: Statistical Reward vs Feature Point Count Reward
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['Statistical Reward'], df['Feature Point Count Reward'], alpha=0.7, color='b')
plt.title('Statistical Reward vs Feature Point Count Reward')
plt.xlabel('Statistical Reward')
plt.ylabel('Feature Point Count Reward')
plt.grid(True)

# Create the second scatter plot: Statistical Reward vs Object Detection Reward
plt.subplot(1, 2, 2)
plt.scatter(df['Statistical Reward'], df['Object Detection Reward'], alpha=0.7, color='g')
plt.title('Statistical Reward vs Object Detection Reward')
plt.xlabel('Statistical Reward')
plt.ylabel('Object Detection Reward')
plt.grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.savefig('./scatter_plots.png')
