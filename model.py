from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset (NEW WORKING ONE)
dataset = load_dataset("tanishqmahajan/resume-dataset")

# Convert to pandas
df = pd.DataFrame(dataset['train'])

# Check column names
print(df.columns)

# Adjust columns if needed
X = df['Resume']   # text
y = df['Category'] # labels

# Convert text to vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")