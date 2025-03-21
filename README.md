# AI-Safety System: Toxic Content Detection

## Overview
This project uses natural language processing (NLP) to identify and mitigate harmful or toxic online content. It focuses on AI safety by training a BERT-based classifier to detect toxicity in user-generated comments. The goal is to ensure responsible language model deployment and moderation systems by flagging identity attacks, threats, and hate speech at scale.

## Tech Stack Used:
**Python:** pandas, numpy, scikit-learn
**NLP:** BERT via Hugging Face Transformers
**Machine Learning:** PyTorch
**API Deployment:** FastAPI (in progress)
**Cloud Hosting:** Render (in progress)

## Dataset
I use the **Jigsaw Specialized Rater Pools Dataset** from [Kaggle](https://www.kaggle.com/datasets/google/jigsaw-specialized-rater-pools-dataset/data).  
The dataset contains **toxic and non-toxic comments**, labeled for identity attacks, insults, threats, and obscenity.

## Features
**Text Preprocessing Pipeline:** Cleaning, lowercasing, punctuation removal
**Binary Classification:** Toxic vs. non-toxic labels using BERT
**Training on Google Colab:** with GPU acceleration (A100)
**Evaluation Metrics:** Accuracy, precision, recall, F1-score
**API Endpoint (in progress)**: Real-time comment classification with FastAPI
**Model Deployment (planned):** To be hosted on Render or Hugging Face Spaces

## Performance
The fine-tuned BERT model achieved:

  -Accuracy: 99.9%
  -F1-score (toxic class): 0.9997
  -Macro average F1-score: 0.9850
These metrics indicate high robustness, especially in differentiating between clean and harmful content.

## Download the Trained Model
The full set of fine-tuned model files is available for download:

Google Drive Folder: https://drive.google.com/drive/folders/1uLslf5BDwLqoB26UGNpvIZf6wK9vy_Fl?usp=sharing

This includes the trained weights, tokenizer configuration, and supporting files needed for inference.

To use it:

Download and unzip the folder.
Place it in your project root as bert_toxic_classifier/.
Run src/api.py or any inference script to load the model.
