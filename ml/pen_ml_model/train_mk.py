import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load CSVs
dys_df = pd.read_csv(r"database\dysgraphia.csv")
normal_df = pd.read_csv(r"database\normal.csv")
dys_df["label"] = 1
normal_df["label"] = 0
df = pd.concat([dys_df, normal_df], ignore_index=True)
df.rename(columns={"p1":"pressure1","p2":"pressure2"}, inplace=True)
df = df.sort_values("timestamp")

# Extract features (1-sec = 100 samples)
batch_size = 100
batches = []
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    if len(batch) < batch_size: continue
    feat = {}
    for c in ["ax","ay","az","gx","gy","gz","pressure1","pressure2"]:
        feat[f"{c}_mean"] = batch[c].mean()
        feat[f"{c}_std"] = batch[c].std()
        feat[f"{c}_min"] = batch[c].min()
        feat[f"{c}_max"] = batch[c].max()
        feat[f"{c}_rms"] = np.sqrt(np.mean(batch[c]**2))
    feat["label"] = batch["label"].mode()[0]
    batches.append(feat)

features_df = pd.DataFrame(batches)

# Train/test split
X = features_df.drop("label", axis=1)
y = features_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Model training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Normal","Dysgraphia"], zero_division=0))
