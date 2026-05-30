import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "data/raw/CAPSTON/dos_filtered_network_traffic.csv"  # Update with your correct file path
df = pd.read_csv(file_path)

# Filter dataset to keep only DoS and Benign traffic
df_dos = df[df['Attack'].isin(['dos', 'benign'])].copy()

# Encode labels: 0 for Benign, 1 for DoS
df_dos['Attack_Label'] = df_dos['Attack'].map({'benign': 0, 'dos': 1})

# Drop non-numeric and unnecessary columns (e.g., IP addresses)
df_dos = df_dos.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack'])

# Normalize numeric features
scaler = StandardScaler()
numeric_columns = df_dos.drop(columns=['Attack_Label']).columns
df_dos[numeric_columns] = scaler.fit_transform(df_dos[numeric_columns])

# Save the preprocessed data
df_dos.to_csv("data/raw/CAPSTON/preprocessed_dos_dataset.csv", index=False)

# Display dataset summary
print(df_dos.head())
print("\nDataset Shape:", df_dos.shape)

