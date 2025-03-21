import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model_path = "bert_toxic_classifier"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify(text):
    """Classify a comment as toxic or non-toxic with confidence score."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label].item()

    return "Toxic" if label == 1 else "Non-Toxic", round(confidence * 100, 2)

if __name__ == "__main__":
    comment = input("Enter a comment to classify: ")
    label, confidence = classify(comment)
    print(f"\nPrediction: {label} ({confidence}% confidence)")