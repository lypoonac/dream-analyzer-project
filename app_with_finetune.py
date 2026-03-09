import streamlit as st
import pandas as pd
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments

# Toggle for fine-tuning (False for Streamlit Cloud deployment; True locally with Python 3.10)
FINE_TUNE_MODE = False

# Fine-tuning function (uses base_zhou_gong.csv)
def fine_tune_model():
    st.write("Starting fine-tuning on Python 3.10...")
    
    df = pd.read_csv('data/base_zhou_gong.csv')
    
    stress_levels = ['Low', 'Medium', 'High']
    rec_templates = ["Practice mindfulness.", "Seek support.", "Exercise to reduce anxiety."]
    
    combined_data = []
    for _, row in df.iterrows():
        stress = random.choice(stress_levels)
        rec = random.choice(rec_templates)
        prompt = f"Dream: {row['dream_text']} Interpretation: {row['interpretation']} Stress: {stress} Recommendation: {rec}"
        combined_data.append({'prompt': prompt})
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('combined_training_data.csv', index=False)
    
    dataset = load_dataset('csv', data_files='combined_training_data.csv')
    dataset = dataset['train'].train_test_split(test_size=0.2)
    
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', clean_up_tokenization_spaces=False)  # Fixes invertibility warning from issue #31884
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    
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
    
    st.write("Fine-tuning complete.")

if FINE_TUNE_MODE:
    fine_tune_model()

# Cache models
@st.cache_resource
def load_generation_model():
    path = 'fine_tuned_combined' if FINE_TUNE_MODE else 'distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=False)  # Fixes issue #31884
    model = AutoModelForCausalLM.from_pretrained(path)
    return tokenizer, model

@st.cache_resource
def load_stress_pipeline():
    return pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

# Load
gen_tokenizer, gen_model = load_generation_model()
stress_classifier = load_stress_pipeline()

# Pipeline 1: Stress detection
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

# Pipeline 2: Interpretation + recommendation (fine-tuned or public)
def generate_interp_rec(dream_text, stress_level):
    if not dream_text.strip():
        return "Invalid input.", "Please provide a dream description."
    prompt = f"Interpret dream in Zhou Gong style: {dream_text}. Stress: {stress_level}. Recommendation:"
    inputs = gen_tokenizer(prompt, return_tensors='pt')
    outputs = gen_model.generate(**inputs, max_length=200, temperature=0.7)
    generated = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = generated.split("Recommendation:")
    interp = parts[0].strip() if parts else "Unable to interpret."
    rec = parts[1].strip() if len(parts) > 1 else "No recommendation."
    return interp, rec

# UI
st.title("Dream Analyzer App (Python 3.10)")
st.write("Enter your dream for analysis.")

dream_input = st.text_area("Describe your dream:", height=150)
if st.button("Analyze"):
    if dream_input.strip():
        with st.spinner("Analyzing..."):
            stress_level = detect_stress(dream_input)
            interpretation, recommendation = generate_interp_rec(dream_input, stress_level)
            
            st.subheader("Stress Level")
            st.write(stress_level)
            
            st.subheader("Interpretation")
            st.write(interpretation)
            
            st.subheader("Recommendation")
            st.write(recommendation)
    else:
        st.error("Enter a dream.")
