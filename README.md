# AI-Safety System: Toxic Content Detection

## Overview  
This project uses natural language processing (NLP) to detect and mitigate toxic online content. It focuses on AI safety by training a BERT-based classifier to identify identity attacks, threats, obscenity, and hate speech in user-generated comments. The goal is to support responsible language model deployment and content moderation at scale.

## Tech Stack  
- **Python**: pandas, numpy, scikit-learn  
- **Machine Learning**: PyTorch  
- **NLP**: BERT via Hugging Face Transformers  
- **Deployment**: FastAPI *(in progress)*  
- **Cloud Hosting**: Render *(planned)*  

## Dataset  
The model is trained on the [Jigsaw Specialized Rater Pools Dataset](https://www.kaggle.com/datasets/google/jigsaw-specialized-rater-pools-dataset/data), which includes comments labeled for identity attacks, insults, threats, and other forms of toxicity.

## Features  
- Clean preprocessing pipeline: lowercasing, punctuation removal, etc.  
- Binary classification: toxic vs. non-toxic comments  
- Trained on Google Colab using A100 GPU  
- Evaluation with standard metrics: accuracy, precision, recall, F1-score  
- Predictive inference via CLI or API (API in progress)

## Model Performance  
The fine-tuned BERT model achieved:

- **Accuracy**: 99.9%  
- **F1-score (Toxic class)**: 0.9997  
- **Macro average F1-score**: 0.9850  

These results reflect strong model performance on the dataset. However, it's worth noting that the training data is highly imbalanced, with a significant skew toward toxic examples. As a result, the model sometimes over-predicts toxicity — flagging neutral or even positive comments (e.g. "You are kind", "You are cute") as toxic with high confidence.

This behavior highlights a common challenge in applied NLP: balancing high precision with real-world fairness and nuance. Future work includes retraining on a balanced dataset and introducing thresholding to reduce false positives for non-toxic language.

## Making Predictions

### Run Locally  
Ensure the model folder is named `bert_toxic_classifier/` and placed in your project root. Then run:

```bash
python src/predict.py --text "You're a horrible person and nobody likes you."
```

Sample output:

```
Step 1: Input Text -> You're a horrible person and nobody likes you.
Step 2: Prediction: Toxic (Confidence: 99.2%)
```

### Run in Google Colab  
1. Mount Google Drive  
2. Upload the full project folder including the model directory  
3. In a code cell, run:

```python
!python src/predict.py --text "Your comment here"
```

This simulates a real-world moderation use case — useful for platforms, chat systems, or backend screening tools.

## Download the Trained Model  
Due to GitHub’s file size limits, the model is hosted externally:

📁 [Download from Google Drive](https://drive.google.com/drive/folders/1uLslf5BDwLqoB26UGNpvIZf6wK9vy_Fl?usp=sharing)

**What’s included:**  
- Model weights (`pytorch_model.bin`, `model.safetensors`)  
- Tokenizer files  
- Configs for BERT-based inference

**To use:**  
- Download and unzip the folder  
- Place it in your root directory as `bert_toxic_classifier/`  
- Run the prediction script or plug into an API
