import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load processed dataset
df = pd.read_csv("data/processed_toxic_comments.csv")

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

# Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["cleaned_text"].tolist(), df["toxic"].tolist(), test_size=0.2, random_state=42
)

# Create dataset objects
train_dataset = ToxicDataset(train_texts, train_labels)
val_dataset = ToxicDataset(val_texts, val_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
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