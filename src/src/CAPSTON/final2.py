import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the preprocessed dataset
df = pd.read_csv(r"data/raw/CAPSTON/preprocessed_dos_dataset.csv")

# Define features and target
features = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES',
            'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']
X = df[features]
y = df['Attack_Label']

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

# Evaluate each model
model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_pred))

    model_scores[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUROC': roc_auc_score(y_test, y_prob)
    }

# Tabular display
results_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={'index': 'Model'})
print("\nModel Evaluation Results:\n")
print(results_df.to_string(index=False))

# Plot results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
models_list = list(model_scores.keys())
scores = {metric: [model_scores[model][metric] for model in models_list] for metric in metrics}

x = np.arange(len(models_list))
width = 0.15

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, scores[metric], width=width, label=metric)

plt.xticks(x + width * 1.5, models_list)
plt.ylabel("Score")
plt.title("Model Performance on Preprocessed DoS Dataset")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("assets/screenshots/CAPSTON/dos_model_comparison_all_models.png", dpi=300, bbox_inches='tight')
plt.show()

