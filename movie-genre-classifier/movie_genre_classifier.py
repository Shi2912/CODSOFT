import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving models

# Download NLTK data (run once)
nltk.download('stopwords')

# Load data
train_df = pd.read_csv(
    'archive/Movie_Classification_Genre/train_data.txt',
    sep=' ::: ',
    engine='python',
    names=['id', 'title', 'genre', 'description']
)

test_df = pd.read_csv(
    'archive/Movie_Classification_Genre/test_data.txt',
    sep=' ::: ',
    engine='python',
    names=['id', 'title', 'description']
)

print("Train data sample:")
print(train_df.head())
print("\nTest data sample:")
print(test_df.head())

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Text preprocessing: lowercase, remove punctuation, stopwords, and stem
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
train_df['clean_desc'] = train_df['description'].apply(preprocess_text)
test_df['clean_desc'] = test_df['description'].apply(preprocess_text)

# TF-IDF Vectorizer with unigrams and bigrams
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

X = tfidf.fit_transform(train_df['clean_desc'])
y = train_df['genre']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
params = {'alpha': [0.1, 0.5, 1.0]}
nb = MultinomialNB()
grid = GridSearchCV(nb, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best alpha: {grid.best_params_['alpha']}")
model = grid.best_estimator_

# Validation performance
y_pred = model.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'movie_genre_nb_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

# Predict on test set and save
X_test = tfidf.transform(test_df['clean_desc'])
test_preds = model.predict(X_test)
test_df['predicted_genre'] = test_preds

test_df[['id', 'title', 'predicted_genre']].to_csv('test_predictions.csv', index=False)

print("\nTest predictions saved to 'test_predictions.csv'")
# Interactive prediction loop
print("\nYou can now enter a movie description to predict its genre (type 'exit' to quit):")

while True:
    user_input = input("\nEnter movie plot summary: ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    # Preprocess input like training data
    processed_input = preprocess_text(user_input)
    vectorized_input = tfidf.transform([processed_input])

    # Predict genre
    pred_genre = model.predict(vectorized_input)[0]

    print(f"Predicted Genre: {pred_genre}")
