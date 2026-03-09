import streamlit as st
import pandas as pd
import random
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments

# Toggle for fine-tuning (Set to True locally to train and save model; False for Streamlit deployment)
FINE_TUNE_MODE = True  # Change to True for local fine-tuning

# Get absolute path to repo root (works on local and Streamlit Cloud)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Fine-tuning function (saves to local folder) - unchanged from previous
def fine_tune_model():
    st.write("Starting fine-tuning locally...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Step 1: Loading and preparing dataset...")
    progress_bar.progress(10)
    
    # Load dataset with error check
    csv_path = os.path.join(REPO_ROOT, 'data/base_zhou_gong.csv')
    if not os.path.exists(csv_path):
        st.error(f"Dataset file '{csv_path}' not found! Add it to the repo.")
        return
    
    df = pd.read_csv(csv_path)
    if 'dream_text' not in df.columns or 'interpretation' not in df.columns:
        st.error("CSV must have 'dream_text' and 'interpretation' columns!")
        return
    
    # Synthetic data
    stress_levels = ['Low', 'Medium', 'High']
    rec_templates = [
        "Practice mindfulness to maintain calm.",
        "Seek support from friends or professionals if stress persists.",
        "Engage in exercise or hobbies to reduce anxiety."
    ]
    
    combined_data = []
    for i, (_, row) in enumerate(df.iterrows()):
        stress = random.choice(stress_levels)
        rec = random.choice(rec_templates)
        prompt = f"Dream: {row['dream_text']} Interpretation: {row['interpretation']} Stress: {stress} Recommendation: {rec}"
        combined_data.append({'prompt': prompt})
        # Update progress every 10%
        if i % max(1, len(df) // 10) == 0:
            progress_bar.progress(10 + int(20 * (i / len(df))))
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(os.path.join(REPO_ROOT, 'combined_training_data.csv'), index=False)
    
    status_text.text("Step 2: Loading dataset for training...")
    progress_bar.progress(30)
    
    dataset = load_dataset('csv', data_files=os.path.join(REPO_ROOT, 'combined_training_data.csv'))
    if len(dataset['train']) == 0:
        st.error("Generated dataset is empty! Check CSV data.")
        return
    dataset = dataset['train'].train_test_split(test_size=0.2)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    status_text.text("Step 3: Tokenizing dataset...")
    progress_bar.progress(40)
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=256)
        tokenized['labels'] = tokenized['input_ids'].copy()  # Add labels for causal LM loss
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Check if labels are present
    if 'labels' not in tokenized_dataset['train'].column_names:
        st.error("Labels not added to dataset! Fine-tuning cannot proceed.")
        return
    
    # Model
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    
    # Training args
    training_args = TrainingArguments(
        output_dir=os.path.join(REPO_ROOT, 'results'),
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
    
    status_text.text("Step 4: Training model (this may take 10-30 minutes)...")
    progress_bar.progress(50)
    
    trainer.train()
    
    status_text.text("Step 5: Saving model...")
    progress_bar.progress(90)
    
    # Save locally (commit this folder to GitHub for deployment)
    trainer.save_model(os.path.join(REPO_ROOT, 'fine_tuned_combined'))
    tokenizer.save_pretrained(os.path.join(REPO_ROOT, 'fine_tuned_combined'))
    
    progress_bar.progress(100)
    status_text.text("Fine-tuning complete. Model saved to 'fine_tuned_combined' folder. Commit to GitHub.")
    st.success("Done! Now set FINE_TUNE_MODE=False and deploy.")

# Run fine-tuning if enabled (only locally)
if FINE_TUNE_MODE:
    fine_tune_model()

# Global variables for models (loaded on demand)
gen_tokenizer = None
gen_model = None
stress_classifier = None

# Cache for generation model
@st.cache_resource
def load_generation_model_cached():
    if FINE_TUNE_MODE:
        path = 'distilgpt2'  # Base model during training
    else:
        path = os.path.join(REPO_ROOT, 'fine_tuned_combined')  # Saved model for deployment
        if not os.path.exists(path):
            st.error(f"Fine-tuned model folder '{path}' not found! Commit it to GitHub.")
            return None, None
    
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=False)  # Fixes issue #31884
    model = AutoModelForCausalLM.from_pretrained(path)
    return tokenizer, model

# Cache for stress pipeline
@st.cache_resource
def load_stress_pipeline_cached():
    return pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

# Function to load all models with spinner
def load_all_models():
    global gen_tokenizer, gen_model, stress_classifier
    with st.spinner("Loading models (this may take a minute the first time)..."):
        gen_tokenizer, gen_model = load_generation_model_cached()
        stress_classifier = load_stress_pipeline_cached()
    if gen_tokenizer is None or gen_model is None:
        st.error("Failed to load generation model!")
        return False
    st.success("Models loaded successfully!")
    return True

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

# Button to load models (avoids blocking startup)
if st.button("Load Models (Required First Time)"):
    load_all_models()

dream_input = st.text_area("Describe your dream:", height=150)
if st.button("Analyze Dream"):
    if dream_input.strip():
        if gen_tokenizer is None or gen_model is None or stress_classifier is None:
            st.error("Models not loaded! Click 'Load Models' first.")
        else:
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
