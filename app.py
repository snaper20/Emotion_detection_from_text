import streamlit as st
try:
    import joblib
except ImportError:
    raise ImportError("Joblib is not installed. Please add 'joblib' to requirements.txt.")

# Load model and vectorizer
model = joblib.load("svm_emotion_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Manual label mapping
label_map = {
    0: 'sadness😢',
    1: 'joy😊',
    2: 'love❤️',
    3: 'anger😠',
    4: 'fear😨',
    5: 'surprise😲'
}

# Streamlit App
st.title("Emotion Detection from Text")
st.write("Enter a sentence and the model will predict the emotion.")

# User Input
user_input = st.text_area("Your Text", "")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        text_vector = vectorizer.transform([user_input])

        # Predict class number
        predicted_label = model.predict(text_vector)[0]

        # Map to emotion name
        predicted_emotion = label_map.get(predicted_label, "Unknown")

        st.success(f"Predicted Emotion: **{predicted_emotion}**")
