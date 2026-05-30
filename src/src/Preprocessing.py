
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("data/raw/CAPSTON/network_traffic.csv")

# 1. Filter for DoS and Benign traffic only
df = df[df['Attack'].isin(['DoS', 'Benign'])]

# 2. Drop irrelevant columns
df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], inplace=True)

# 3. Encode target: 1 = DoS, 0 = Benign
df['Label'] = df['Attack'].apply(lambda x: 1 if x == 'DoS' else 0)

# 4. Select features
features = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
            'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']

X = df[features]
y = df['Label']

# 5. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Display result summary
print("Preprocessing Complete.")
print("Shape of input data:", X_scaled.shape)
print("\nClass Distribution:")
print(df['Label'].value_counts().rename(index={0: 'Benign', 1: 'DoS'}))


