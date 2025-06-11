import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load and preprocess the dataset
def load_data(filepath='spam.csv'):
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']] 
    df.columns = ['label', 'message']  
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels: ham=0, spam=1
    return df

# Step 2: Split the dataset
def split_data(df):
    X = df['message']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocess text with TF-IDF
def preprocess_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Step 4: Train and evaluate Naive Bayes
def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nNaive Bayes Performance:")
    print_evaluation(y_test, predictions)
    return model

# Step 5: Train and evaluate Logistic Regression
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nLogistic Regression Performance:")
    print_evaluation(y_test, predictions)
    return model

# Step 6: Train and evaluate Support Vector Machine
def train_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nSupport Vector Machine Performance:")
    print_evaluation(y_test, predictions)
    return model

# Step 7: Print evaluation metrics
def print_evaluation(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Step 8: Main function to run everything
def main():
    print("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = preprocess_text(X_train, X_test)

    print("\nTraining Naive Bayes...")
    train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test)

    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf, y_test)

    print("\nTraining Support Vector Machine...")
    train_svm(X_train_tfidf, y_train, X_test_tfidf, y_test)

    # Interactive Prediction with Logistic Regression
    print("\n--- Test Your Own Message ---")
    user_msg = input("Enter a message to classify as spam or ham: ")
    user_input_tfidf = vectorizer.transform([user_msg])
    prediction = lr_model.predict(user_input_tfidf)
    print("\nPrediction:", "Spam" if prediction[0] == 1 else "Ham (Not Spam)")

# Entry point
if __name__ == "__main__":
    main()
