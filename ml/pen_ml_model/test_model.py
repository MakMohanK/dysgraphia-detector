# ---------- test_model.py ----------
import pandas as pd
import numpy as np
import joblib

# ==== Load trained model ====
model = joblib.load("dysgraphia_model.pkl")
print("✅ Model loaded successfully.")

# ==== Load new data (replace with your test CSV) ====
test_df = pd.read_csv(r"database\test_data.csv")

test_df.rename(columns={"p1":"pressure1","p2":"pressure2"}, inplace=True)
test_df = test_df.sort_values("timestamp")

# ==== Feature Extraction ====
batch_size = 100   # must match training
batches = []
for i in range(0, len(test_df), batch_size):
    batch = test_df.iloc[i:i+batch_size]
    if len(batch) < batch_size:
        continue
    features = {}
    for col in ["ax","ay","az","gx","gy","gz","pressure1","pressure2"]:
        features[f"{col}_mean"] = batch[col].mean()
        features[f"{col}_std"] = batch[col].std()
        features[f"{col}_min"] = batch[col].min()
        features[f"{col}_max"] = batch[col].max()
        features[f"{col}_rms"] = np.sqrt(np.mean(batch[col]**2))
    batches.append(features)

features_df = pd.DataFrame(batches)

# ==== Predict ====
predictions = model.predict(features_df)
labels = ["Normal" if p == 0 else "Dysgraphia" for p in predictions]

for i, label in enumerate(labels):
    print(f"Batch {i+1}: {label}")
