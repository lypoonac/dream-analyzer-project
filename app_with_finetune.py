import streamlit as st
import pandas as pd
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments

# Toggle for fine-tuning (set to True locally to train, False for deployment/Streamlit Cloud)
FINE_TUNE_MODE = False  # Change to True for local fine-tuning

# Fine-tuning function (combines interpretation + recommendation using base_zhou_gong.csv)
def fine_tune_model():
    st.write("Starting fine-tuning...")
    
    # Load dataset
    df = pd.read_csv('data/base_zhou_gong.csv')  # Your CSV in data/
    
    # Synthetic data for combined prompts
    stress_levels = ['Low', 'Medium', 'High']
    rec_templates = [
        "Practice mindfulness to maintain calm.",
        "Seek support from friends or professionals if stress persists.",
        "Engage in exercise or hobbies to reduce anxiety."
    ]
    combined_data = []
    for _, row in df.iterrows():
        stress = random.choice(stress_levels)
        rec = random.choice(rec_templates)
        prompt = f"Dream: {row['dream_text']} Interpretation: {row['interpretation']} Stress: {stress} Recommendation: {rec}"
        combined_data.append({'prompt': prompt})
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('combined_training_data.csv', index=False)
    
    # Load for training
    dataset = load_dataset('csv', data_files='combined_training_data.csv')
    dataset = dataset['train'].train_test_split(test_size=0.2)
    
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')  # Public base
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')  # Public base
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        evaluation_strategy='epoch',
        save_steps=100,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )
    
    trainer.train()
    trainer.save_model('fine_tuned_combined')
    tokenizer.save_pretrained('fine_tuned_combined')
    
    st.write("Fine-tuning complete. Model saved locally.")

# Run fine-tuning if mode is enabled (skipped on Streamlit Cloud)
if FINE_TUNE_MODE:
    fine_tune_model()

# Cache for app performance
@st.cache_resource
def load_generation_model():
    try:
        # Load fine-tuned if available (local dev), else public
        tokenizer = AutoTokenizer.from_pretrained('fine_tuned_combined' if FINE_TUNE_MODE else 'distilgpt2')
        model = AutoModelForCausalLM.from_pretrained('fine_tuned_combined' if FINE_TUNE_MODE else 'distilgpt2')
    except:
        # Fallback to public if error
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    return tokenizer, model

@st.cache_resource
def load_stress_pipeline():
    return pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')  # Public

# Load
gen_tokenizer, gen_model = load_generation_model()
stress_classifier = load_stress_pipeline()

# Pipeline 1: Stress detection (public)
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

# Pipeline 2: Combined interp + rec (fine-tuned or public with prompting)
def generate_interp_rec(dream_text, stress_level):
    if not dream_text.strip():
        return "Invalid input.", "Please provide a dream description."
    prompt = (
        f"Interpret this dream using Zhou Gong's traditional Chinese method: {dream_text}. "
        f"Provide a symbolic interpretation. Then, based on stress level {stress_level}, give a personalized recommendation. "
        f"Format: Interpretation: [text] Recommendation: [text]"
    )
    inputs = gen_tokenizer(prompt, return_tensors='pt')
    outputs = gen_model.generate(**inputs, max_length=200, num_return_sequences=1, temperature=0.7, top_p=0.9)
    generated = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = generated.split("Recommendation:")
    interp = parts[0].replace("Interpretation:", "").strip() if parts else "Unable to interpret."
    rec = parts[1].strip() if len(parts) > 1 else "No recommendation generated."
    return interp, rec

# Streamlit UI (the app part)
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
                st.error(f"Error: {str(e)}. Try again.")
    else:
        st.error("Please enter a dream description.")
