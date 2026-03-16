# Bank-Customer-Churn-Prediction
This project builds a machine learning model to predict customer churn

# Project Workflow

## 1. Data Loading

import pandas as pd

df = pd.read_csv("Churn.csv")

## 2. Data Cleaning

Some columns were removed because they do not contribute to prediction.

RowNumber

CustomerId

Surname

df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)

## 3. Checking Missing Values

Before modeling, the dataset was checked for missing values.

df.isnull().sum()

Result:

No missing values were found in the dataset.

This allowed the modeling process to continue without imputation.

## 4. Encode Categorical Variables

Machine learning algorithms require numerical input.

The categorical columns Geography and Gender were converted into numeric format using one-hot encoding.

df = pd.get_dummies(df, drop_first=True)

This creates new columns such as:

Geography_Germany

Geography_Spain

Gender_Male

## 5. Exploratory Data Analysis (EDA)

Exploratory analysis helps understand patterns in the dataset and relationships between variables.

## Churn Distribution

Understanding how many customers churned versus stayed.

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Distribution")
plt.show()

<img width="745" height="592" alt="Churn1" src="https://github.com/user-attachments/assets/ce101b4d-8692-4e68-8685-600a0b8ccf07" />

Observation:

Most customers did not churn

The dataset is slightly imbalanced

## Age vs Churn

Analyzing whether age affects customer churn.

sns.boxplot(x="Exited", y="Age", data=df)
plt.title("Age Distribution by Churn")
plt.show()

<img width="722" height="587" alt="ChunrAge" src="https://github.com/user-attachments/assets/d34aedea-639d-4c8e-a5f4-64b758ad3011" />

Observation:

Customers who churn tend to be older on average than those who stay.

## Correlation Heatmap

A correlation heatmap helps identify relationships between variables.

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.show()

<img width="849" height="736" alt="churnheatmap" src="https://github.com/user-attachments/assets/b5e2aefa-c40e-4e9a-a2eb-8c13c68cbf12" />

Insights

Age shows a positive correlation with churn

Active members are less likely to churn

Balance and number of products show moderate relationships

## 6. Feature / Target Split

X = df.drop("Exited", axis=1)
y = df["Exited"]

## 7. Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

## 8. Feature Scaling

Feature scaling improves model convergence and performance.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## 9. Model Training

A Logistic Regression model was used for classification.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

## 10. Model Evaluation

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

# Results

Model performance:

Accuracy: ~80–83%

The model successfully predicts customer churn using customer financial and demographic features.

# Technologies Used

Python

Pandas

NumPy

Seaborn

Matplotlib

Scikit-Learn

Jupyter Notebook

# Future Improvements

Possible enhancements include:

Confusion Matrix

ROC Curve and AUC Score

Feature Importance Analysis

# Author

Janay Wesso

Data Science / Machine Learning Portfolio Project
