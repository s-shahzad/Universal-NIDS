import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/raw/CAPSTON/network_traffic.csv")  # Ensure this path is correct

# Count each attack type
attack_counts = df['Attack'].value_counts()

# Plot the bar chart
plt.figure(figsize=(10, 6))
attack_counts.plot(kind='bar', color='teal')

# Customize the chart
plt.title("Distribution of Attack Types in Network Traffic Dataset")
plt.xlabel("Attack Type")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()

# Show the plot
plt.show()

