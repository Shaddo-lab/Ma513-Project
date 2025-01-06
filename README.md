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
```

## Dataset Format

The dataset is structured in JSONLINES format, where each line represents a JSON object containing the following fields:

- `tokens`: A list of words or tokens in a sentence.
- `ner_tags`: A list of NER labels for each token (note: these are not present in the test dataset).
- `unique_id`: A unique identifier for each sentence (only in the test dataset).

Here is an example of what a line in the dataset looks like:

```json
{
  "tokens": ["The", "company", "Apple", "is", "based", "in", "California"],
  "ner_tags": ["O", "O", "B-Entity", "O", "O", "O", "B-Location"]
}
```
## How to Use

To train the model, follow these steps:

1. **Prepare the Dataset**: Place the training and validation datasets in your project directory, naming them `NER-TRAINING.jsonlines`,`NER-TESTING.jsonlines` and `NER-VALIDATION.jsonlines`.
   
3. **Execute the  Script**: Run the code with all the files in the same repertory.
   
5. **Evaluation**: After running the file, you will have an evaluation of the model on the validation dataset, and it wsill print you a classification report.
   
6. **Output file**:  A prediction file named : `NER-TESTING-PREDICTIONS.jsonlines` will be created in the same repertory. 



