import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📄 Resume Screening AI")

st.write("Upload or paste your resume text below:")

resume_text = st.text_area("Enter Resume Text")

if st.button("Predict Category"):
    if resume_text:
        data = vectorizer.transform([resume_text])
        prediction = model.predict(data)[0]
        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please enter resume text")