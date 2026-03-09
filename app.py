import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

# Load fine-tuned models (upload these to Hugging Face for production)
interpreter_tokenizer = AutoTokenizer.from_pretrained('fine_tuned_dream_interpreter')  # Or your HF repo
interpreter_model = AutoModelForCausalLM.from_pretrained('fine_tuned_dream_interpreter')

recommender_tokenizer = AutoTokenizer.from_pretrained('fine_tuned_recommender')
recommender_model = AutoModelForCausalLM.from_pretrained('fine_tuned_recommender')

# Stress detection model (pre-trained)
stress_classifier = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

# Function to interpret dream using Zhou Gong (generate text)
def interpret_dream(dream_text):
    prompt = f"Interpret this dream using Zhou Gong's method: {dream_text}. Interpretation:"
    inputs = interpreter_tokenizer(prompt, return_tensors='pt')
    outputs = interpreter_model.generate(**inputs, max_length=100, num_return_sequences=1)
    return interpreter_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to detect stress level
def detect_stress(dream_text):
    result = stress_classifier(dream_text)[0]
    emotion = result['label']
    if emotion in ['fear', 'sadness']:  # Map to stress
        return 'High'
    elif emotion in ['anger', 'disgust']:
        return 'Medium'
    else:
        return 'Low'

# Function to generate recommendation
def generate_recommendation(interpretation, stress_level):
    prompt = f"Based on dream interpretation: {interpretation}. Stress level: {stress_level}. Personalized recommendation:"
    inputs = recommender_tokenizer(prompt, return_tensors='pt')
    outputs = recommender_model.generate(**inputs, max_length=150, num_return_sequences=1)
    return recommender_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Dream Analyzer Business App")
st.write("Enter your dream description below (in English). We'll interpret it using Zhou Gong's method, assess stress, and provide recommendations.")

dream_input = st.text_area("Describe your dream:")
if st.button("Analyze Dream"):
    if dream_input:
        with st.spinner("Analyzing..."):
            interpretation = interpret_dream(dream_input)
            stress_level = detect_stress(dream_input)
            recommendation = generate_recommendation(interpretation, stress_level)
        
        st.subheader("Zhou Gong Interpretation")
        st.write(interpretation)
        
        st.subheader("Estimated Stress Level")
        st.write(stress_level)
        
        st.subheader("Personalized Recommendation")
        st.write(recommendation)
    else:
        st.error("Please enter a dream description.")

# For validation/testing: Add sample dreams and expected outputs in your project report
