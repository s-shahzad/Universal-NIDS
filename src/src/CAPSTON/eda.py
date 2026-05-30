# Step 2: Exploratory Data Analysis (EDA)

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
file_path = r"data/raw/CAPSTON/datasheet2.xlsx"
df = pd.read_excel(file_path)

# Data Cleaning: Drop rows with missing values
df_cleaned = df.dropna()

# Display the first few rows to ensure data is loaded correctly
df_cleaned.head(), df_cleaned.info()

# 1. Label Distribution Analysis
label_counts = df_cleaned['string.4'].value_counts()

# Plot Label Distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 2. Port Distribution Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['port.1'], bins=50, color='blue')
plt.title('Port Distribution')
plt.xlabel('Port')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# 3. Packet Count Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['count.3'], bins=50, color='green')
plt.title('Packet Count Distribution')
plt.xlabel('Packet Count')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# 4. Protocol Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='enum', data=df_cleaned, palette='viridis')
plt.title('Protocol Type Distribution')
plt.xlabel('Protocol Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Final refinement of DDoS attack labeling with a focus on broader criteria and high traffic volume

# Initialize 'attack_type' column with 'Benign' as default
df_cleaned['attack_type'] = 'Benign'

# Further broaden criteria and reduce thresholds for DDoS attacks
# HTTP Flood: Lower threshold for port 80 (HTTP traffic)
df_cleaned.loc[(df_cleaned['port.1'] == 80) & (df_cleaned['count.3'] > 1), 'attack_type'] = 'HTTP Flood'

# SYN Flood: Generalized to all high-volume TCP traffic
df_cleaned.loc[(df_cleaned['enum'] == 'tcp') & (df_cleaned['count.3'] > 1), 'attack_type'] = 'SYN Flood'

# UDP Flood: Generalized to all high-volume UDP traffic
df_cleaned.loc[(df_cleaned['enum'] == 'udp') & (df_cleaned['count.3'] > 1), 'attack_type'] = 'UDP Flood'

# DNS Amplification: Broaden criteria for DNS traffic on port 53
df_cleaned.loc[(df_cleaned['port.1'] == 53) & (df_cleaned['count.3'] > 1), 'attack_type'] = 'DNS Amplification'

# NTP Amplification: Lower threshold for port 123 (NTP traffic)
df_cleaned.loc[(df_cleaned['port.1'] == 123) & (df_cleaned['count.3'] > 1), 'attack_type'] = 'NTP Amplification'

# IRC-based Attacks: Generalized criteria for IRC traffic on port 6667
df_cleaned.loc[(df_cleaned['port.1'] == 6667) & (df_cleaned['count.3'] > 1), 'attack_type'] = 'IRC Flood'

# Visualize Updated Distribution of Attack Types
final_attack_counts = df_cleaned['attack_type'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=final_attack_counts.index, y=final_attack_counts.values, palette='viridis')
plt.title('Final Refined Distribution of Attack Types')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Display first few rows to verify the new labels
df_cleaned[['string.4', 'attack_type', 'port.1', 'count.3', 'enum']].head()

