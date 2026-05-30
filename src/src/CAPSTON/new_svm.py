
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split, RandomizedSearchCV
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.svm                import SVC
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)


data_path = 'data/raw/CAPSTON/network_traffic.csv'
df = pd.read_csv(data_path)


df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack'])


X = df.drop(columns=['Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_tune, _, y_tune, _ = train_test_split(
    X_train_scaled, y_train,
    train_size=0.2,
    stratify=y_train,
    random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),    # no-op since X_tune already scaled, but kept for pipeline integrity
    ('svm',    SVC(probability=True, random_state=42))
])

param_dist = {
    'svm__C':     [0.1, 1.0, 10.0],
    'svm__gamma': ['scale', 'auto', 0.01],
    'svm__kernel':['rbf']
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=5,            # 5 random trials
    cv=2,                # 2-fold CV
    scoring='f1',        # optimize F1
    n_jobs=1,            # single-core to avoid resource issues
    verbose=2,
    random_state=42
)

print("\n Running hyperparameter search on 20% subsample...")
t0 = time.time()
search.fit(X_tune, y_tune)
t1 = time.time()

print(" Best params   :", search.best_params_)
print(f" Tuning time    : {t1 - t0:.2f} sec")

best_svm = search.best_estimator_

t2 = time.time()
best_svm.fit(X_train_scaled, y_train)
t3 = time.time()
print(f" Full-train time: {t3 - t2:.2f} sec")

t4 = time.time()
y_pred  = best_svm.predict(X_test_scaled)
t5 = time.time()

y_proba = best_svm.predict_proba(X_test_scaled)[:,1]

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_proba)

print("\n Model: Support Vector Machine (Optimized)")
print(f"Training Time : {t3 - t2:.2f} sec")
print(f"Testing Time  : {t5 - t4:.2f} sec")
print(f"Accuracy      : {acc:.4f}")
print(f"Precision     : {prec:.4f}")
print(f"Recall        : {rec:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"AUROC         : {auroc:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign","DoS"])
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap='Blues', values_format=',')
plt.title("SVM (Optimized) â€” Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

