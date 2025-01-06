# Cybersecurity Named Entity Recognition (NER)

## Overview
This project focuses on applying Named Entity Recognition (NER) for cybersecurity-related text. The system is built to classify cybersecurity entities such as malware, attack vectors, and other technical terms using a specialized NER model. The data used for training, validation, and testing is annotated in the IOB2 format.

## Key Features
- Classify cybersecurity-related entities using token-level classification.
- Preprocess text and tokenize sentences using Hugging Face's Transformers library.
- Train models with GPU support to speed up the process.
- Generate predictions in JSONLINES format for easy integration.
- Evaluate model performance through standard metrics like F1-score.

## Requirements
Ensure you are using Python 3.7 or higher, and install the following dependencies:

```bash
pip install torch transformers datasets pandas scikit-learn
