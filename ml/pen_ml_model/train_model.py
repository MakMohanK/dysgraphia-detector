# ---------- train_model.py ----------
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==== Load datasets ====
dys_df = pd.read_csv(r"database\dysgraphia.csv")
normal_df = pd.read_csv(r"database\normal.csv")
dys_df["label"] = 1
normal_df["label"] = 0

# Combine and preprocess
df = pd.concat([dys_df, normal_df], ignore_index=True)
df.rename(columns={"p1":"pressure1","p2":"pressure2"}, inplace=True)
df = df.sort_values("timestamp")

# ==== Feature Extraction ====
batch_size = 100   # 1-second windows (assuming ~100 Hz sampling)
batches = []
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    if len(batch) < batch_size:
        continue
    features = {}
    for col in ["ax","ay","az","gx","gy","gz","pressure1","pressure2"]:
        features[f"{col}_mean"] = batch[col].mean()
        features[f"{col}_std"] = batch[col].std()
        features[f"{col}_min"] = batch[col].min()
        features[f"{col}_max"] = batch[col].max()
        features[f"{col}_rms"] = np.sqrt(np.mean(batch[col]**2))
    features["label"] = batch["label"].mode()[0]
    batches.append(features)

features_df = pd.DataFrame(batches)

# ==== Train/Test Split ====
X = features_df.drop("label", axis=1)
y = features_df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==== Train Model ====
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==== Evaluate ====
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Normal", "Dysgraphia"], zero_division=0))

# ==== Save Model ====
joblib.dump(model, "dysgraphia_model.pkl")
print("\n✅ Model saved as dysgraphia_model.pkl")
