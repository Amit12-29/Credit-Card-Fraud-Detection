import os

os.makedirs("screenshots", exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/data/creditcard.csv")

print("Dataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['Class'].value_counts())

print("\nStatistical Summary:")
print(df.describe())


plt.figure(figsize=(8,5))

class_counts = df['Class'].value_counts()

bars = plt.bar(
    ['Genuine', 'Fraud'],
    class_counts.values
)

plt.yscale('log')

plt.title("Credit Card Transaction Distribution (Log Scale)")
plt.xlabel("Transaction Type")
plt.ylabel("Count (Log Scale)")

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f'{int(y):,}',
        ha='center'
    )

plt.tight_layout()

plt.savefig("screenshots/class_distribution.png")
plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Genuine", "Fraud"]
)

disp.plot()

plt.title("Confusion Matrix")
plt.savefig("screenshots/confusion_matrix.png")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("screenshots/roc_curve.png")
plt.show()

import joblib

joblib.dump(model, "fraud_model.pkl")

print("\nModel Saved Successfully!")