import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('new_data.csv')

# Split dataset into features and labels
X = data['instruction']
y = data['response']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TfidfVectorizer with optimizations
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)

# Initialize LinearSVC with optimal parameters
svm_classifier = LinearSVC(dual=False)

# Create and train pipeline
pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('clf', svm_classifier)])
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save trained model
joblib.dump(pipeline, 'trained_model.joblib')

# Load model once
model = joblib.load('trained_model.joblib')

# Function to get response
def get_response(instruction):
    return model.predict([instruction])[0]

# Example usage
instruction = "I want to track my order."
print(get_response(instruction))
