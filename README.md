# Credit Card Fraud Detection Using Machine Learning

## Project Overview

Credit card fraud is a major challenge in digital payment systems. This project uses Machine Learning to classify credit card transactions as genuine or fraudulent.

A Logistic Regression model is trained on historical transaction data. The project also includes data visualization, model evaluation, and sample-based prediction testing.

## Objectives

* Detect potentially fraudulent credit card transactions.
* Analyze transaction patterns using data visualization.
* Evaluate model performance using standard classification metrics.
* Demonstrate how Machine Learning can support fraud detection.

## Dataset

This project uses the **Credit Card Fraud Detection** dataset, commonly available through Kaggle.

**Dataset source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymized credit card transactions made by European cardholders over two days in September 2013.

### Dataset Statistics

| Property                |                                    Value |
| ----------------------- | ---------------------------------------: |
| Total transactions      |                                  284,807 |
| Genuine transactions    |                                  284,315 |
| Fraudulent transactions |                                      492 |
| Total features          | 30 input features plus the target column |

### Features

* `Time`: Seconds elapsed between a transaction and the first transaction in the dataset.
* `V1` to `V28`: Anonymized features obtained using Principal Component Analysis (PCA).
* `Amount`: Transaction amount.
* `Class`: Target variable.

Target labels:

* `0`: Genuine transaction
* `1`: Fraudulent transaction

### Downloading the Dataset

The dataset CSV is not included in this repository because of its large file size.

1. Open the Kaggle dataset page linked above.

2. Download the dataset and extract the CSV file.

3. Place `creditcard.csv` in the following location:

   `dataset/data/creditcard.csv`

4. Install the project dependencies using the instructions below.

**Note:** Verify the dataset source against the file used in your experiments before citing it in academic submissions.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

## Machine Learning Model

**Algorithm:** Logistic Regression

The model is evaluated using:

* Accuracy
* Confusion Matrix
* Receiver Operating Characteristic (ROC) Curve
* Area Under the ROC Curve (AUC)

Because fraudulent transactions are rare, accuracy alone does not fully describe fraud detection performance. Precision, recall, and F1-score are also useful metrics when evaluating this type of imbalanced classification problem.

## Model Performance

The following results are reported by the project. They should be interpreted according to the dataset split and evaluation procedure used during experimentation.

| Metric    | Reported Value |
| --------- | -------------: |
| Accuracy  |         99.93% |
| AUC Score |         0.9509 |

### Confusion Matrix

|                   | Predicted Genuine | Predicted Fraudulent |
| ----------------- | ----------------: | -------------------: |
| Actual Genuine    |       56,850 (TN) |              14 (FP) |
| Actual Fraudulent |           28 (FN) |              70 (TP) |

* **True Negatives (TN):** 56,850 genuine transactions correctly classified.
* **False Positives (FP):** 14 genuine transactions incorrectly classified as fraudulent.
* **False Negatives (FN):** 28 fraudulent transactions incorrectly classified as genuine.
* **True Positives (TP):** 70 fraudulent transactions correctly classified.

These values are reproduced from the project's existing results. Confirm that the confusion matrix and reported accuracy were calculated from the same evaluation split before using them in a formal report.

## Visualizations

The project includes the following visualizations:

1. Credit card transaction class distribution.
2. Confusion matrix.
3. ROC curve.

The generated figures are available in the `screenshots/` directory.

## Testing

The model was tested using transaction samples covering different prediction outcomes:

* Genuine transactions.
* Fraudulent transactions.
* True Positive cases.
* True Negative cases.
* False Positive cases.
* False Negative cases.

A 20-transaction demonstration was also performed to inspect the model's predictions across different scenarios.

## Project Structure

```text
Credit_Card_Fraud_Detection/
├── dataset/
│   └── data/
│       └── creditcard.csv       # Download separately
├── screenshots/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── src/                         # Source code
├── report/                      # Project report
├── fraud_model.pkl               # Trained model
├── test.csv                      # Test data
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Amit12-29/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the CSV from the Kaggle dataset page and place it at:

```text
dataset/data/creditcard.csv
```

### 5. Run the project

Run the relevant Python script from the `src/` directory. The exact script name and command depend on the files included in that directory.

## Future Improvements

* Compare Logistic Regression with Random Forest and XGBoost.
* Investigate class-imbalance techniques such as SMOTE.
* Report precision, recall, F1-score, and precision-recall AUC.
* Tune the classification threshold according to fraud detection requirements.
* Develop a web interface for interactive predictions.
* Explore real-time fraud detection workflows.

## Author

**Amit Singh**
Computer Science Engineering Student
Machine Learning & Automation Enthusiast

## License

This project is intended for educational and internship purposes. Refer to the `LICENSE` file in this repository for the applicable license terms.
