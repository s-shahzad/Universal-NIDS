import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
file_path = "data/raw/CAPSTON/network_traffic.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Convert IP addresses to numerical values (IPV4 to Integer)
df['IPV4_SRC_ADDR'] = df['IPV4_SRC_ADDR'].apply(lambda ip: int(ip.replace(".", "")))
df['IPV4_DST_ADDR'] = df['IPV4_DST_ADDR'].apply(lambda ip: int(ip.replace(".", "")))

# Encoding 'Attack' column (Benign = 0, Malicious = 1)
label_encoder = LabelEncoder()
df['Attack'] = label_encoder.fit_transform(df['Attack'])

# Feature Scaling (Normalizing numerical values)
scaler = MinMaxScaler()
numerical_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 
                  'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 
                  'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])



# Check class distribution (Benign vs Malicious)
plt.figure(figsize=(6, 4))
df['Attack'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.xticks(ticks=[0, 1], labels=['Benign', 'Malicious'], rotation=0)
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.title("Class Distribution: Benign vs Malicious")
plt.show()

# Check unique values in the 'Attack' column to identify malicious categories
malicious_types = df["Attack"].unique()

# Count occurrences of each attack type
malicious_counts = df["Attack"].value_counts()

# Display the results
malicious_types, malicious_counts


# Save processed data
df.to_csv("data/raw/CAPSTON/processed_network_traffic.csv", index=False)

print("Data preprocessing complete! Processed file saved as 'data/raw/CAPSTON/processed_network_traffic.csv'.")

