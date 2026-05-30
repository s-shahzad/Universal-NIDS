import pandas as pd

# Load preprocessed dataset
file_path = "data/raw/CAPSTON/processed_network_traffic.csv"
df = pd.read_csv(file_path)

# Check if the 'Attack' column contains numeric values
print("Unique Attack Values:", df["Attack"].unique())

# Define mapping (Modify this if needed)
attack_mapping = {
    '0': 'benign',
    '1': 'exploits',
    '2': 'fuzzers',
    '3': 'reconnaissance',
    '4': 'generic',
    '5': 'dos',  # This should match the correct label for DoS
    '6': 'analysis',
    '7': 'backdoor',
    '8': 'shellcode',
    '9': 'worms'
}

# Convert 'Attack' column to string and map it back to text labels
df["Attack"] = df["Attack"].astype(str).map(attack_mapping)

# Debugging Step: Check unique attack types again
print("Mapped Attack Types:", df["Attack"].unique())

# Now filter only 'dos' and 'benign' samples
df_dos = df[df["Attack"] == "dos"]
df_benign = df[df["Attack"] == "benign"]

# Debugging Step: Check dataset size
print(f"Number of DoS samples: {len(df_dos)}")
print(f"Number of Benign samples: {len(df_benign)}")

# Save the corrected dataset
output_path = "data/raw/CAPSTON/dos_filtered_network_traffic.csv"
df.to_csv(output_path, index=False)

print(f"\nâœ… Fixed dataset saved as '{output_path}'.")

