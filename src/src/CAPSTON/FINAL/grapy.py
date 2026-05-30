import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick


model_results = {
    "Decision Tree": {
        "Accuracy": 0.9988,
        "Precision": 0.8146,
        "Recall": 0.8267,
        "F1 Score": 0.8206,
        "AUROC": 0.9155,
        "Train Time": 22.92,
        "Test Time": 0.15
    },
    "Random Forest": {
        "Accuracy": 0.9992,
        "Precision": 0.8755,
        "Recall": 0.8703,
        "F1 Score": 0.8729,
        "AUROC": 0.9978,
        "Train Time": 279.47,
        "Test Time": 3.96
    },
    "XGBoost": {
        "Accuracy": 0.9986,
        "Precision": 0.7975,
        "Recall": 0.7604,
        "F1 Score": 0.7785,
        "AUROC": 0.9683,
        "Train Time": 2.25,
        "Test Time": 0.16
    }
}


df_results = pd.DataFrame(model_results).T.reset_index().rename(columns={'index': 'Model'})


sns.set(style="whitegrid")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.13
x = np.arange(len(df_results['Model']))

for i, metric in enumerate(metrics):
    ax.bar(x + i * bar_width, df_results[metric], width=bar_width, label=metric)

ax.set_xticks(x + (len(metrics) - 1) * bar_width / 2)
ax.set_xticklabels(df_results['Model'])
ax.set_ylim(0.7, 1.05)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Model Performance Comparison (Excluding SVM)")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()


plt.savefig(r"data/raw/CAPSTON/FINAL/model_performance_comparison.png", dpi=300)
plt.show()

