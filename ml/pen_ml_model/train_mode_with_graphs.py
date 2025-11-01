# train_model_with_graphs.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# ====== Load Data ======
dys = pd.read_csv(r"database\dysgraphia.csv")
normal = pd.read_csv(r"database\normal.csv")

dys["label"] = 1
normal["label"] = 0
df = pd.concat([dys, normal], ignore_index=True)

# ---- Check column names ----
print("Available columns:", list(df.columns))

# Update these if needed to match your dataset headers
sensor_columns = ["ax", "ay", "az", "gx", "gy", "gz", "p1", "p2"]

# ====== Feature Extraction ======
batch_size = 100  # 1-sec batch approx
batches = []
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    if len(batch) < batch_size:
        continue
    features = {}
    for col in sensor_columns:
        if col not in batch.columns:
            continue
        features[f"{col}_mean"] = batch[col].mean()
        features[f"{col}_std"] = batch[col].std()
        features[f"{col}_min"] = batch[col].min()
        features[f"{col}_max"] = batch[col].max()
        features[f"{col}_rms"] = np.sqrt(np.mean(batch[col]**2))
    features["label"] = batch["label"].mode()[0]
    batches.append(features)

features_df = pd.DataFrame(batches)
print(f"Extracted features shape: {features_df.shape}")

# ====== Train/Test Split ======
X = features_df.drop("label", axis=1)
y = features_df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ====== Train Model ======
n_trees = 200
model = RandomForestClassifier(
    n_estimators=n_trees, random_state=42, warm_start=True
)

train_acc_list = []
val_acc_list = []

for i in range(1, n_trees + 1):
    model.n_estimators = i
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_test)
    train_acc_list.append(accuracy_score(y_train, y_train_pred))
    val_acc_list.append(accuracy_score(y_test, y_val_pred))

# ====== Final Evaluation ======
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"\nFinal Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["Normal", "Dysgraphia"], zero_division=0))

# ====== Accuracy Curve ======
plt.figure(figsize=(8,5))
plt.plot(train_acc_list, label="Training Accuracy", color="blue")
plt.plot(val_acc_list, label="Validation Accuracy", color="orange")
plt.title("Training vs Validation Accuracy (Random Forest)")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.show()

# ====== Confusion Matrix ======
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Normal", "Dysgraphia"],
            yticklabels=["Normal", "Dysgraphia"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ====== Feature Importance ======
plt.figure(figsize=(10,6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ====== Save Model ======
joblib.dump(model, "dysgraphia_model.pkl")
print("✅ Model saved as dysgraphia_model.pkl")
