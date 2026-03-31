# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Create Sample Dataset (works without external URL)
data = pd.DataFrame({
    'ResumeText': [
        "Experienced Python developer with Machine Learning skills",
        "Fresher looking for entry-level software job",
        "Expert in data analysis and AI projects",
        "High school graduate with no programming experience",
        "Worked on multiple ML and AI projects with Python",
        "Seeking internship, basic computer knowledge"
    ],
    'Label': [
        'selected',
        'not selected',
        'selected',
        'not selected',
        'selected',
        'not selected'
    ]
})

print("Sample Dataset:")
print(data)

# Step 3: Encode Labels
data['Label'] = data['Label'].map({'not selected': 0, 'selected': 1})

# Step 4: Split Data
X = data['ResumeText']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert Text to Numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test_vec)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Continuous User Input for Screening
print("\nResume Screening AI – Enter resumes to predict. Type 'exit' to quit.\n")

while True:
    resume_text = input("Paste resume text:\n")
    
    if resume_text.strip().lower() == "exit":
        print("Exiting Resume Screening AI. Goodbye!")
        break
    
    # Convert resume text to vector
    resume_vec = vectorizer.transform([resume_text])
    
    # Predict
    prediction = model.predict(resume_vec)
    
    # Output result
    if prediction[0] == 1:
        print("Prediction: Selected ✅\n")
    else:
        print("Prediction: Not Selected ❌\n")