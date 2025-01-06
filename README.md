# Named Entity Recognition (NER) for Cybersecurity

## Groupe 12: Paul-Louis Cremadeills, Pierre Samoyault  
IPSA, 63 Bd de Brandebourg  
Ma513 – Hands-on Machine Learning for Cybersecurity, 2024  

## Abstract
Named Entity Recognition (NER) is a key technique in Natural Language Processing (NLP) that extracts structured information from unstructured text by identifying and classifying entities such as names, organizations, and technical terms. In the context of cybersecurity, NER is essential for analyzing reports, detecting threats, and categorizing critical components such as malware and attack vectors. This project focuses on building a cybersecurity-specific NER system utilizing data from the SemEval-2018 Task 8 competition. The aim is to identify and classify domain-specific entities using the IOB2 tagging format.  

Several transformer-based models, including BERT-Base-Cased, SecBERT (a cybersecurity-optimized version of BERT), and Large-BERT-Uncased, were evaluated for their effectiveness in recognizing cybersecurity-related entities. The models' performance was assessed using the F1-score, which revealed the trade-offs between domain specialization and general-purpose model capabilities. The results indicate that although SecBERT excels in handling technical cybersecurity terms, Large-BERT-Uncased offers greater versatility and precision, especially with a larger vocabulary, provided sufficient computational resources are available for training.

## Introduction
Named Entity Recognition (NER) is a fundamental Natural Language Processing (NLP) task that focuses on extracting valuable information from raw text by identifying specific entities, such as people, organizations, or specialized terms. In the field of cybersecurity, NER becomes crucial for processing and analyzing technical documents, identifying cyber threats, and categorizing key elements like malware or attack techniques. By automating the extraction of such entities, NER aids in enhancing the efficiency of threat detection and response.

This project aims to develop an NER system specifically tailored to the cybersecurity domain, utilizing data from the SemEval-2018 Task 8 competition. The goal is to classify relevant cybersecurity entities into the IOB2 format, a standard used in NER tasks to label token positions within entities. The effectiveness of this system is assessed through the F1-score, providing insights into the challenges and performance trade-offs of adapting general NLP models to specialized cybersecurity contexts.

## Data Description
The dataset used in this project is structured in JSON Lines format, where each line represents a dictionary containing the following keys:
- **unique_id**: A unique identifier for each sentence.
- **tokens**: A list of words forming the sentence.
- **ner_tags**: A list of NER labels corresponding to each word, using the IOB2 format. “B-“ for the beginning of an entity and “I-“ for the inside of an entity.

Available files include `NER-TRAINING.jsonlines`, `NER-VALIDATION.jsonlines`, and `NER-TESTING.jsonlines`. The validation file is used for final evaluation, while the training and testing files are used to optimize model performance and prepare the dataset. The dataset shows a significant imbalance with "0" tags being more prevalent than other tags, indicating the importance of balancing tag weights for model performance.

## Data Preparation
To ensure the model can efficiently process inputs and produce reliable predictions, the following steps were performed during the data preparation phase:

- **Loading and Exploration**: JSON Lines files were loaded and explored to understand the distribution of labels. This basic exploration confirmed the presence of key entities such as "Action," "Entity," and "Modifier."
- **Label Mapping**: NER text labels were converted into numeric indices for compatibility with classification models. The bidirectional mapping (label2id and id2label) facilitates both training and result interpretation.
- **Tokenization and Alignment**: Sentences were tokenized into sub-tokens using a transformer-based tokenizer. NER labels were aligned with these sub-tokens to ensure correct label assignment during training.

## Models and Results
For this project, three main models were experimented with: 

1. **BERT-Base-Cased**: A robust baseline model with strong generalization across diverse tasks, but less specialized for cybersecurity-specific vocabulary and concepts.
2. **SecBERT**: A fine-tuned version of BERT specialized for cybersecurity, offering better performance on technical entities but prone to overfitting for rare entities.
3. **Large-BERT-Uncased**: A general-purpose model with a broader vocabulary, useful for handling rare or varied terms, though computationally expensive.

The BERT-Base model provided consistent performance but struggled with cybersecurity-specific terms, achieving a precision of only 1.28% for 30 epochs. SecBERT showed strong results for domain-specific tasks but faced limitations for recognizing less frequent entities. The Large-BERT-Uncased model, even with fewer epochs (5 epochs), demonstrated better precision and versatility, particularly for rare terms.

## Conclusion
This project demonstrates the development of a Named Entity Recognition (NER) system tailored to the cybersecurity domain, leveraging advanced transformer-based models such as BERT-Base-Cased, SecBERT, and Large-BERT-Uncased. The findings reveal that while specialized models like SecBERT provide a strong starting point for cybersecurity tasks, general-purpose models such as Large-BERT-Uncased can offer superior results with adequate computational resources. 

An important observation from the dataset analysis was the disparity in the distribution of NER tags, with "0" being significantly more prevalent than other tags. Addressing this imbalance through techniques like re-weighting tag importance or oversampling less frequent tags could enhance model performance. 

Future work could focus on optimizing training time, addressing tag imbalances, and exploring further fine-tuning techniques to improve the robustness of the model.

## Setup Instructions

### Prerequisites:
- Python 3.7 or higher
- Required libraries:
  - `transformers`
  - `torch`
  - `sklearn`
  - `pandas`
  - `numpy`

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ner-cybersecurity.git
