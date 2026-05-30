import pandas as pd

# Load the dataset
file_path = r"data/raw/CAPSTON/network_traffic.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
df_info = df.info()
df_head = df.head()

# Check for missing values
missing_values = df.isnull().sum()

# Display the results
df_info, df_head, missing_values


