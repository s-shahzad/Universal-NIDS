# plot_confusion_matrices.py

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    # 1) Load & preprocess
    df = pd.read_csv("data/raw/CAPSTON/network_traffic.csv")
    df = df[df['Attack'].isin(['DoS','Benign'])]
    df.drop(columns=['IPV4_SRC_ADDR','IPV4_DST_ADDR'], inplace=True)
    df['Label'] = df['Attack'].map({'Benign':0,'DoS':1})

    features = [
        'L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO',
        'IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS',
        'TCP_FLAGS','FLOW_DURATION_MILLISECONDS'
    ]
    X = df[features]
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2) Single train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Define & train models (DT, RF, XGBoost)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric='logloss', n_jobs=-1, random_state=42
        )
    }

    # 4) Fit, predict & collect confusion matrices
    cms = {}
    for name, mdl in models.items():
        print(f"Training {name}â€¦")
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        cms[name] = confusion_matrix(y_te, y_pred, labels=[0,1])

    # 5) Plot side-by-side confusion matrices
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","DoS"])
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

