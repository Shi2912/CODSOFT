# customer-churn.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# ---------------------- Data Exploration ----------------------
print("Data Snapshot:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Exited', palette='Set2')
plt.title("Churn Distribution (0 = Not Churned, 1 = Churned)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------------- Data Cleaning ----------------------
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

# One-hot encode Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# ---------------------- Feature Selection ----------------------
X = df.drop('Exited', axis=1)
y = df['Exited']

# ---------------------- Data Scaling ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------- Model Training & Evaluation ----------------------
def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
evaluate(lr, "Logistic Regression")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate(rf, "Random Forest")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
evaluate(gb, "Gradient Boosting")

# ---------------------- Feature Importance (Gradient Boosting) ----------------------
importances = gb.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Gradient Boosting")
plt.barh(range(len(indices)), importances[indices], color='skyblue')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.tight_layout()
plt.show()

# ---------------------- Predict from Custom Input ----------------------
print("\n--- Churn Prediction for a Custom Customer ---")
custom_input = []

# Collect user input (make sure order matches feature order)
try:
    custom_input.append(int(input("Credit Score (e.g. 600): ")))
    custom_input.append(int(input("Gender (0 = Female, 1 = Male): ")))
    custom_input.append(int(input("Age: ")))
    custom_input.append(int(input("Tenure (years with bank): ")))
    custom_input.append(float(input("Balance: ")))
    custom_input.append(int(input("Num of Products: ")))
    custom_input.append(int(input("Has Credit Card? (1 = Yes, 0 = No): ")))
    custom_input.append(int(input("Is Active Member? (1 = Yes, 0 = No): ")))
    custom_input.append(float(input("Estimated Salary: ")))
    custom_input.append(int(input("Geography_Germany? (1 = Yes, 0 = No): ")))
    custom_input.append(int(input("Geography_Spain? (1 = Yes, 0 = No): ")))

    custom_input_scaled = scaler.transform([custom_input])
    prediction = gb.predict(custom_input_scaled)

    if prediction[0] == 1:
        print("⚠️ The customer is likely to churn.")
    else:
        print("✅ The customer is likely to stay.")
except Exception as e:
    print("Invalid input. Please enter values carefully.")
