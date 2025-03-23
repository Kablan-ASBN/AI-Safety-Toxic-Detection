
# AI-Safety-Toxic-Content-Detection  
BERT-based toxic comment classifier for responsible AI safety | Identity attack & hate speech detection | CLI & API-ready NLP project

This system leverages BERT and NLP techniques to detect and flag toxic online content — including identity attacks, threats, obscenity, and hate speech. Designed for AI safety, it demonstrates how machine learning can support responsible language model deployment and large-scale content moderation.

---

## Project Summary  
"Protect users, platforms, and communities from toxic language."

Toxic online content remains a persistent challenge across digital platforms. This project demonstrates the the use of fine-tuned transformer models to classify user-generated comments as toxic or non-toxic. It supports scalable moderation systems and future integrations with language models and safety pipelines.

---

## Dataset

- Source: Jigsaw Specialized Rater Pools Dataset  
  https://www.kaggle.com/datasets/google/jigsaw-specialized-rater-pools-dataset/data
- Labels: Identity attack, threat, obscenity, insult, and toxicity
- Use Case: Binary classification — toxic (1) vs. non-toxic (0)

---

## Key Features

- Clean NLP preprocessing: lowercasing, punctuation stripping, etc.
- Fine-tuned BERT model for toxic comment detection
- CLI-based inference with sample inputs and outputs
- FastAPI-based deployment planned
- High-performance training with A100 GPU on Google Colab
- External model hosting due to file size limits

---

## Tech Stack

- Language: Python  
- Libraries: pandas, numpy, scikit-learn  
- ML Framework: PyTorch  
- NLP: Hugging Face Transformers (BERT)  
- Deployment: FastAPI (in progress)  
- Hosting: Render (planned)

---

## Model Performance

| Metric                 | Score     |
|------------------------|-----------|
| Accuracy               | 99.9%     |
| F1-Score (Toxic Class) | 0.9997    |
| Macro F1-Score         | 0.9850    |

The model was trained on a highly imbalanced dataset skewed toward toxic comments.  
While recall is excellent, it sometimes over-predicts toxicity, flagging neutral comments like "You're kind" as toxic.  
Future improvements include balanced retraining and smarter threshold tuning.

---

## CLI Usage

### Run Locally
```
python src/predict.py --text "You're a horrible person and nobody likes you."
```

Example Output:
```
Step 1: Input Text -> You're a horrible person and nobody likes you.
Step 2: Prediction: Toxic (Confidence: 99.2%)
```

### Run in Google Colab
```
!python src/predict.py --text "Your comment here"
```

Requires uploaded model folder and mounted Google Drive.

---

## Model Download

Due to GitHub’s file size limits, the model is hosted externally:

Download from Google Drive:  
https://drive.google.com/drive/folders/1uLslf5BDwLqoB26UGNpvIZf6wK9vy_Fl?usp=sharing

Included:
- pytorch_model.bin, model.safetensors  
- Tokenizer files  
- BERT config files

Usage:
- Place the unzipped folder as bert_toxic_classifier/ in your project root
- Run predictions or deploy the model via API

---

## Deployment (Planned)

- FastAPI REST endpoint: /predict  
- Input: JSON with a comment string  
- Output: Toxic or non-toxic with confidence  
- Hosting target: Render or Hugging Face Spaces

---

## Project Structure

```
ai-safety-toxic-content-detection/
├── src/
│   ├── predict.py           # CLI inference script
│   └── deploy/              # (Planned) FastAPI app
├── bert_toxic_classifier/   # Downloaded model directory
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Lessons Learned

- How to fine-tune BERT for sensitive classification tasks
- The impact of dataset imbalance on real-world NLP models
- The trade-off between high recall and false positives
- How to prepare an NLP model for production with APIs and CLI tools

---

## Contact

G. Kablan Assebian  
MSc Data Science Candidate | NLP Enthusiast | Aspiring ML Engineer  
gomis.k.assebian@gmail.com
