import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load("fraud_model.pkl")

# Predictions
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nTRUE NEGATIVE (Actual=0, Predicted=0)")
print(results[(results["Actual"] == 0) & (results["Predicted"] == 0)].head(5))

print("\nTRUE POSITIVE (Actual=1, Predicted=1)")
print(results[(results["Actual"] == 1) & (results["Predicted"] == 1)].head(5))

print("\nFALSE POSITIVE (Actual=0, Predicted=1)")
print(results[(results["Actual"] == 0) & (results["Predicted"] == 1)].head(5))

print("\nFALSE NEGATIVE (Actual=1, Predicted=0)")
print(results[(results["Actual"] == 1) & (results["Predicted"] == 0)].head(5))