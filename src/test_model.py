import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load("fraud_model.pkl")

# Predict
y_pred = model.predict(X_test)

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred

# Categories
TN = results[(results["Actual"] == 0) & (results["Predicted"] == 0)].head(5)
TP = results[(results["Actual"] == 1) & (results["Predicted"] == 1)].head(5)
FP = results[(results["Actual"] == 0) & (results["Predicted"] == 1)].head(5)
FN = results[(results["Actual"] == 1) & (results["Predicted"] == 0)].head(5)

demo = pd.concat([TN, TP, FP, FN])

print("=" * 90)
print("CREDIT CARD FRAUD DETECTION - 20 TRANSACTION DEMO")
print("=" * 90)

for i, (_, row) in enumerate(demo.iterrows(), start=1):

    features = row.drop(["Actual", "Predicted"]).to_frame().T

    fraud_prob = model.predict_proba(features)[0][1]

    print(
        f"{i:02d}. Actual={int(row['Actual'])} | "
        f"Predicted={int(row['Predicted'])} | "
        f"Fraud Probability={fraud_prob*100:.4f}%"
    )