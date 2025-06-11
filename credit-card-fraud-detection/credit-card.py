import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('fraudTrain.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'fraudTrain.csv' not found.")
    print("Please ensure 'fraudTrain.csv' is directly in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    exit()


# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Dataset Head (first 5 rows) ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

if 'is_fraud' not in df.columns:
    print("\nError: 'is_fraud' column not found in the loaded dataset.")
    print("Please check your CSV file and ensure it has a column named 'is_fraud' (case-sensitive).")
    exit()

print("\n--- Class Distribution (Legitimate vs. Fraudulent) ---")
print(df['is_fraud'].value_counts())
print(df['is_fraud'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x='is_fraud', data=df)
plt.title('Class Distribution (0: Legitimate, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Number of Transactions')
plt.xticks(ticks=[0, 1], labels=['Legitimate (0)', 'Fraudulent (1)'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# --- NEW: Handle non-numeric and irrelevant columns ---
columns_to_drop = []

if 'Unnamed: 0' in X.columns:
    columns_to_drop.append('Unnamed: 0')

for col in X.columns:
    if X[col].dtype == 'object' and col not in columns_to_drop:
        columns_to_drop.append(col)

if 'unix_time' in X.columns and 'unix_time' not in columns_to_drop:
    columns_to_drop.append('unix_time')
if 'trans_num' in X.columns and 'trans_num' not in columns_to_drop:
    columns_to_drop.append('trans_num')

X = X.drop(columns=columns_to_drop, errors='ignore')
print(f"\nDropped non-numeric or irrelevant columns: {columns_to_drop}")


# --- 3. Data Preprocessing: Scaling 'amt' ---
if 'amt' in X.columns:
    scaler = StandardScaler()
    X['amt'] = scaler.fit_transform(X[['amt']])
    print("Scaled 'amt' column.")
else:
    print("Error: 'amt' column not found in the dataset. Cannot perform scaling.")
    exit()

print("\n--- Features after handling non-numeric columns and scaling 'amt' ---")
print(X.head())
print(f"Features remaining: {X.columns.tolist()}")


# --- 4. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True) * 100}")
print(f"Testing target distribution:\n{y_test.value_counts(normalize=True) * 100}")

# --- 5. Address Class Imbalance using SMOTE ---
print("\n--- Applying SMOTE to the training data ---")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training target distribution:\n{y_train.value_counts()}")
print(f"Resampled training target distribution:\n{y_train_resampled.value_counts()}")

# --- 6. Model Training and Evaluation ---

def evaluate_model(model, X_test_data, y_test_data, model_name):
    """
    Trains the model and evaluates its performance on the test set.
    Prints classification report, confusion matrix, and key metrics.
    Returns the trained model for later use.
    """
    print(f"\n--- Evaluating {model_name} ---")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_data)

    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test_data, y_pred))

    cm = confusion_matrix(y_test_data, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Predicted Legitimate', 'Predicted Fraud'],
                yticklabels=['Actual Legitimate', 'Actual Fraud'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 50)
    return model # Return the trained model


trained_logistic_model = LogisticRegression(solver='liblinear', random_state=42)
trained_decision_tree_model = DecisionTreeClassifier(max_depth=15, random_state=42)


# --- Evaluate and train models ---
trained_logistic_model = evaluate_model(trained_logistic_model, X_test, y_test, "Logistic Regression")
trained_decision_tree_model = evaluate_model(trained_decision_tree_model, X_test, y_test, "Decision Tree")


print("\n--- Model Training and Evaluation Complete ---")
print("Remember that for fraud detection, 'Recall' (the ability to capture all fraudulent transactions) is often more critical than 'Precision' (avoiding false positives).")
print("However, a balance between both is usually desired depending on the business's tolerance for false positives.")


# --- 7. Interactive Test Section ---
print("\n" + "="*70)
print("             Interactive Fraud Detection Test")
print("="*70)

# --- Select a model for testing ---

model_for_test = trained_decision_tree_model
model_name_for_test = "Decision Tree"

print(f"Using the trained **{model_name_for_test}** model for demonstration.")
print("We'll select a transaction from the test set and predict if it's fraudulent.")

while True:
    try:
        test_index = int(input(f"\nEnter an index number (0 to {len(X_test) - 1}) from the test set to predict fraud: "))
        if 0 <= test_index < len(X_test):
            break
        else:
            print("Invalid index. Please enter a number within the valid range.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

selected_transaction_features = X_test.iloc[[test_index]]
actual_label = y_test.iloc[test_index]

# Make a prediction
predicted_label = model_for_test.predict(selected_transaction_features)[0]

print("\n--- Selected Transaction Details (Preprocessed Features) ---")
print(selected_transaction_features)

print(f"\n--- Prediction Result ---")
print(f"Selected Transaction Index: {test_index}")
print(f"Actual Label: {'Fraudulent (1)' if actual_label == 1 else 'Legitimate (0)'}")
print(f"Predicted Label: {'Fraudulent (1)' if predicted_label == 1 else 'Legitimate (0)'}")

if predicted_label == actual_label:
    print("Result: **CORRECTLY CLASSIFIED!**")
    if predicted_label == 1:
        print("This fraudulent transaction was successfully identified.")
    else:
        print("This legitimate transaction was correctly identified.")
else:
    print("Result: **MISCLASSIFIED!**")
    if predicted_label == 1:
        print("This legitimate transaction was incorrectly flagged as fraudulent (False Positive).")
    else:
        print("This fraudulent transaction was missed (False Negative).")

print("="*70)

