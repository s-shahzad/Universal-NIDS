

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the full dataset
df = pd.read_csv('data/raw/CAPSTON/network_traffic.csv')

# Drop non-numeric and non-essential columns
df_numeric = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack'])

# Features and target
X = df_numeric.drop(columns=['Label'])
y = df_numeric['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm = SVC(probability=True, kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm.predict(X_test_scaled)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'DoS'])

# Plot in contrast style
fig, ax = plt.subplots(figsize=(10, 6))
disp.plot(ax=ax, cmap='Blues', values_format=',')
plt.title("SVM Confusion Matrix (Full Dataset)")
plt.grid(False)
plt.tight_layout()
plt.savefig("assets/screenshots/CAPSTON/svm_full_dataset.png")
plt.show()

