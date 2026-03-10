# Step 1: Install a specific version of transformers to ensure compatibility (run this first)
!pip uninstall -y transformers  # Remove any existing version
!pip install transformers==4.41.2 datasets torch accelerate numpy pandas

# Step 2: Import libraries
import pandas as pd
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
import shutil  # For zipping the folder
import transformers  # To check version
print(f"Transformers version: {transformers.__version__}")

# Fine-tuning function
def fine_tune_model(csv_path='/content/base_zhou_gong.csv', save_path='/content/fine_tuned_combined'):
    print("Starting fine-tuning in Colab...")

    # Load dataset with error check
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file '{csv_path}' not found! Upload it to Colab.")
        return
    
    df = pd.read_csv(csv_path)
    if 'dream_text' not in df.columns or 'interpretation' not in df.columns:
        print("Error: CSV must have 'dream_text' and 'interpretation' columns!")
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
        if i % 50 == 0:
            print(f"Prepared {i}/{len(df)} prompts...")
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('/content/combined_training_data.csv', index=False)
    print("Dataset prepared and saved.")
    
    # Load for training
    dataset = load_dataset('csv', data_files='/content/combined_training_data.csv')
    if len(dataset['train']) == 0:
        print("Error: Generated dataset is empty! Check CSV data.")
        return
    dataset = dataset['train'].train_test_split(test_size=0.2)
    print(f"Dataset split: {len(dataset['train'])} train, {len(dataset['test'])} test examples.")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        tokenized = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=256)
        tokenized['labels'] = tokenized['input_ids'].copy()  # Add labels for causal LM loss
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Check if labels are present
    if 'labels' not in tokenized_dataset['train'].column_names:
        print("Error: Labels not added to dataset! Fine-tuning cannot proceed.")
        return
    
    # Model (ignore unexpected keys like attn.bias, as it's safe for distilgpt2)
    model = AutoModelForCausalLM.from_pretrained('distilgpt2', ignore_mismatched_sizes=True)
    
    # Training args (reduced for speed: smaller batch, 1 epoch for testing)
    # Omitted eval_strategy to avoid version conflicts; defaults to 'no' (no evaluation during training)
    training_args = TrainingArguments(
        output_dir='/content/results',
        num_train_epochs=1,  # Start with 1; increase to 5 once working
        per_device_train_batch_size=2,  # Small for free Colab GPU
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,  # Frequent logging
        fp16=True,  # Enable mixed precision for GPU speed
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )
    
    print("Starting training... (monitor progress below)")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        return
    
    print("Saving model...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Fine-tuning complete. Model saved to 'fine_tuned_combined' folder.")
    
    # Zip the folder for easy download
    shutil.make_archive('/content/fine_tuned_combined', 'zip', save_path)
    print("Zipped folder created: /content/fine_tuned_combined.zip. Download it from Colab Files sidebar.")

# Run the function (upload your CSV first if not already)
fine_tune_model(csv_path='/content/base_zhou_gong.csv')  # Adjust path if your CSV is named differently
