import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Cache for performance on Streamlit Cloud
@st.cache_resource
def load_combined_model():
    tokenizer = AutoTokenizer.from_pretrained('yourusername/fine_tuned_combined_interp_rec')  # Your HF repo (fine-tuned on base_zhou_gong.csv)
    model = AutoModelForCausalLM.from_pretrained('yourusername/fine_tuned_combined_interp_rec')
    return tokenizer, model

@st.cache_resource
def load_stress_pipeline():
    return pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')  # Pre-trained HF pipeline for emotion/stress

# Load models/pipelines
combined_tokenizer, combined_model = load_combined_model()
stress_classifier = load_stress_pipeline()

# Pipeline 1: Detect stress
def detect_stress(dream_text):
    if not dream_text.strip():
        return 'Unknown'
    result = stress_classifier(dream_text)[0]
    emotion = result['label']
    if emotion in ['fear', 'sadness']:
        return 'High'
    elif emotion in ['anger', 'disgust']:
        return 'Medium'
    else:
        return 'Low'

# Pipeline 2: Combined interpretation + recommendation (fine-tuned on base_zhou_gong.csv)
def generate_interp_rec(dream_text, stress_level):
    if not dream_text.strip():
        return "Invalid input.", "Please provide a dream description."
    prompt = f"Dream: {dream_text} Interpretation: [Zhou Gong based] Stress: {stress_level} Recommendation:"
    inputs = combined_tokenizer(prompt, return_tensors='pt')
    outputs = combined_model.generate(**inputs, max_length=200, num_return_sequences=1, temperature=0.7)
    generated = combined_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Interpretation:")[-1].strip()
    parts = generated.split("Recommendation:")
    interp = parts[0].strip() if parts else "Unable to interpret."
    rec = parts[1].strip() if len(parts) > 1 else "No recommendation generated."
    return interp, rec

# Streamlit UI
st.title("Dream Analyzer Business App")
st.write("Enter your dream (in English) for Zhou Gong interpretation, stress level, and personalized recommendation.")

dream_input = st.text_area("Describe your dream:", height=150)
if st.button("Analyze Dream"):
    if dream_input.strip():
        with st.spinner("Analyzing..."):
            try:
                stress_level = detect_stress(dream_input)
                interpretation, recommendation = generate_interp_rec(dream_input, stress_level)
                
                st.subheader("Estimated Stress Level")
                st.write(stress_level)
                
                st.subheader("Zhou Gong Interpretation")
                st.write(interpretation)
                
                st.subheader("Personalized Recommendation")
                st.write(recommendation)
            except Exception as e:
                st.error(f"Error: {str(e)}. Check model paths or try again.")
    else:
        st.error("Please enter a dream description.")
