import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Load processed dataset
df = pd.read_csv("data/processed_toxic_comments.csv")

# Balance dataset using upsampling
toxic_df = df[df["toxic"] == 1]
non_toxic_df = df[df["toxic"] == 0]

# Upsample Non-Toxic to match Toxic count
non_toxic_upsampled = resample(
    non_toxic_df, replace=True, n_samples=len(toxic_df), random_state=42
)

# Combine & shuffle dataset
balanced_df = pd.concat([toxic_df, non_toxic_upsampled]).sample(frac=1, random_state=42)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define PyTorch dataset class
class ToxicDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Split data into train & validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    balanced_df["cleaned_text"].tolist(), 
    balanced_df["toxic"].tolist(), 
    test_size=0.2, 
    random_state=42
)

# Create dataset objects
train_dataset = ToxicDataset(train_texts, train_labels)
val_dataset = ToxicDataset(val_texts, val_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Move model to A100 GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert_model",
    per_device_train_batch_size=32,  # Larger batch size for A100
    per_device_eval_batch_size=32,  
    num_train_epochs=5,  # More epochs since data is balanced
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  # Enable mixed precision for A100
    logging_dir="logs",
    report_to="none"
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("bert_toxic_classifier")
tokenizer.save_pretrained("bert_toxic_classifier")

print("Model training complete! Model saved to 'bert_toxic_classifier'.")