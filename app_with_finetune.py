import streamlit as st
import pandas as pd
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments

# Toggle for fine-tuning (Set to True locally to train and save model; False for Streamlit deployment)
FINE_TUNE_MODE = True  # Change to True for local fine-tuning

# Fine-tuning function (saves to local folder)
def fine_tune_model():
    st.write("Starting fine-tuning locally...")
    
    # Load dataset
    df = pd.read_csv('data/base_zhou_gong.csv')
    
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
    
    # Tokenizer with fix for invertibility warning (issue #31884)
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Model
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    
    # Training args
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
    
    # Save locally (commit this folder to GitHub for deployment)
    trainer.save_model('fine_tuned_combined')
    tokenizer.save_pretrained('fine_tuned_combined')
    
    st.write("Fine-tuning complete. Model saved to 'fine_tuned_combined' folder. Commit this folder to GitHub.")

# Run fine-tuning if enabled (only locally)
if FINE_TUNE_MODE:
    fine_tune_model()

# Cache for performance
@st.cache_resource
def load_generation_model():
    # Load from local saved folder (must be in GitHub repo for deployment)
    path = 'fine_tuned_combined' if not FINE_TUNE_MODE else 'distilgpt2'  # Use saved model for deployment
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=False)  # Fixes issue #31884
    model = AutoModelForCausalLM.from_pretrained(path)
    return tokenizer, model

@st.cache_resource
def load_stress_pipeline():
    return pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

# Load models
gen_tokenizer, gen_model = load_generation_model()
stress_classifier = load_stress_pipeline()

# Pipeline 1: Detect stress (pre-trained HF model)
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

# Pipeline 2: Generate interpretation + recommendation (fine-tuned HF model)
def generate_interp_rec(dream_text, stress_level):
    if not dream_text.strip():
        return "Invalid input.", "Please provide a dream description."
    prompt = f"Interpret this dream using Zhou Gong's method: {dream_text}. Stress level: {stress_level}. Provide interpretation and recommendation.\nInterpretation:\nRecommendation:"
    inputs = gen_tokenizer(prompt, return_tensors='pt')
    outputs = gen_model.generate(**inputs, max_length=200, num_return_sequences=1, temperature=0.8, do_sample=True, top_p=0.95)
    generated = gen_tokenizer.decode(outputs[0], skip_special_tokens=True).split(prompt)[-1].strip()  # Remove prompt from output
    
    # Split and fallback
    if "Recommendation:" in generated:
        parts = generated.split("Recommendation:")
        interp = parts[0].replace("Interpretation:", "").strip()
        rec = parts[1].strip()
    else:
        interp = generated
        rec = "General recommendation: Reflect on emotions and relax."
    
    return interp, rec

# Streamlit UI
st.title("Dream Analyzer Business App")
st.write("Enter your dream description (in English) to get a Zhou Gong interpretation, stress level, and personalized recommendations.")

dream_input = st.text_area("Describe your dream:", height=150)
if st.button("Analyze Dream"):
    if dream_input.strip():
        with st.spinner("Analyzing your dream..."):
            stress_level = detect_stress(dream_input)
            interpretation, recommendation = generate_interp_rec(dream_input, stress_level)
            
            st.subheader("Estimated Stress Level")
            st.write(stress_level)
            
            st.subheader("Zhou Gong Interpretation")
            st.write(interpretation)
            
            st.subheader("Personalized Recommendation")
            st.write(recommendation)
    else:
        st.error("Please enter a dream description.")
