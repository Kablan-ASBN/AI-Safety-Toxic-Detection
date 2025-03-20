import pandas as pd
import re
import string
import os

print("All imports are working!")

def clean_text(text):
    """Remove special characters, extra spaces, and lowercase."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

def preprocess_data(input_file, output_file):
    """Preprocess dataset: clean text, handle missing values, create labels."""
    # Load dataset
    df = pd.read_csv(input_file, encoding="utf-8", quotechar='"')

    # Drop unnecessary columns
    df = df.drop(columns=["id", "unique_contributor_id", "rater_group"])

    # Drop missing values
    df = df.dropna()

    # Create a single `toxic` label (1 if any toxicity label is 1)
    df["toxic"] = df[["identity_attack", "insult", "obscene", "threat", "toxic_score"]].max(axis=1)

    # Drop original toxicity columns
    df = df.drop(columns=["identity_attack", "insult", "obscene", "threat", "toxic_score"])

    # Apply text cleaning function
    df["cleaned_text"] = df["comment_text"].apply(clean_text)

    # Drop the original comment_text column
    df = df.drop(columns=["comment_text"])

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Data preprocessing complete. Saved to {output_file}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data("data/specialized_rater_pools_data.csv", "data/processed_toxic_comments.csv")