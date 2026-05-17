# Multilingual NLP Sentiment Analysis Using DistilBERT

A research-oriented Natural Language Processing (NLP) project that fine-tunes a multilingual transformer model for sentiment analysis using the IMDB movie review dataset. This project demonstrates transformer fine-tuning, multilingual inference, evaluation methodologies, confidence analysis, and qualitative error analysis using the multilingual DistilBERT architecture.

---

## Overview

This project explores multilingual sentiment classification using the pretrained transformer model:

* DistilBERT

The model is fine-tuned on the English IMDB dataset and later evaluated across multiple languages to demonstrate cross-lingual transfer learning capabilities.

The implementation includes:

* Transformer fine-tuning
* Dynamic token padding
* Evaluation metrics
* Confusion matrix visualization
* Classification reports
* Error analysis
* Multilingual inference testing
* Confidence score analysis
* Research-oriented NLP insights

---

## Features

* Fine-tunes multilingual DistilBERT for binary sentiment classification
* Uses the IMDB movie review dataset
* Supports multilingual sentiment inference
* Performs quantitative and qualitative evaluation
* Generates confusion matrix visualizations
* Includes confidence score estimation
* Demonstrates cross-lingual generalization
* Research-focused transformer workflow

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* Matplotlib

---

## Model Architecture

The project uses:

```python
distilbert-base-multilingual-cased
```

This multilingual transformer supports more than 100 languages and enables cross-lingual semantic understanding through shared multilingual embeddings.

---

## Dataset

Dataset used:

* IMDB Movie Reviews Dataset

The dataset contains:

* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment labels:

  * Positive
  * Negative

For faster experimentation, smaller subsets are selected:

```python
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
```

---

## Installation

Install the required dependencies:

```bash
pip uninstall -y transformers accelerate -q

pip install transformers==4.40.1 \
accelerate==0.29.3 \
datasets \
scikit-learn \
matplotlib \
torch -q
```

After installation:

```text
Runtime -> Restart Session
```

Then rerun the notebook cells.

---

## Project Pipeline

### 1. Load Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

### 2. Load Dataset

```python
dataset = load_dataset("imdb")
```

### 3. Tokenization

```python
tokenizer(
    examples["text"],
    truncation=True
)
```

Dynamic padding is handled using:

```python
DataCollatorWithPadding
```

### 4. Load Transformer Model

```python
AutoModelForSequenceClassification
```

### 5. Fine-Tuning

The project uses the Hugging Face Trainer API for supervised fine-tuning.

### 6. Evaluation

Metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 7. Error Analysis

Misclassified reviews are inspected manually for deeper NLP analysis.

### 8. Multilingual Testing

The model is evaluated on:

* English
* French
* Arabic

---

## Training Configuration

```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    fp16=torch.cuda.is_available()
)
```

---

## Evaluation Metrics

The project computes:

* Accuracy
* Precision
* Recall
* F1-score

Using:

```python
precision_recall_fscore_support()
accuracy_score()
classification_report()
```

---

## Multilingual Inference Example

Example multilingual predictions:

### English

```text
"This movie was a brilliant masterpiece."
→ POSITIVE
```

### French

```text
"Ce film était magnifique, j'ai adoré l'histoire."
→ POSITIVE
```

### Arabic

```text
"كان الفيلم سيئاً ومملاً للغاية"
→ NEGATIVE
```

---

## Error Analysis

The project investigates incorrect predictions to identify transformer limitations, including:

* Sarcasm
* Ambiguous sentiment
* Mixed emotional expressions
* Contextual nuance
* Cultural language variation

---

## Confidence Score Analysis

Softmax probabilities are used to estimate prediction confidence:

```python
probs = F.softmax(outputs.logits, dim=1)
```

This provides insight into model certainty and uncertainty estimation.

---

## Research Insights

Key observations from the project:

1. Transformer architectures effectively capture contextual semantic information.
2. Multilingual transformers enable cross-lingual transfer learning.
3. Fine-tuning on English data still allows limited multilingual generalization.
4. Sentiment ambiguity remains challenging for NLP systems.
5. Confidence estimation helps analyze prediction reliability.
6. Low-resource languages and dialects remain difficult research problems.
7. Qualitative error analysis is essential for robust NLP evaluation.

---

## Future Improvements

Potential future work includes:

* Larger multilingual transformers
* XLM-RoBERTa experimentation
* Multilingual fine-tuning datasets
* Explainable AI integration
* Bias and fairness evaluation
* Sarcasm detection
* Low-resource language adaptation
* Domain-specific transfer learning
* Robust multilingual benchmarking

---

## Example Output

The project generates:

* Evaluation metrics
* Classification report
* Confusion matrix visualization
* Misclassification examples
* Multilingual predictions
* Confidence scores

---

## Learning Outcomes

This project demonstrates practical understanding of:

* Transformer architectures
* Multilingual NLP
* Hugging Face ecosystem
* Fine-tuning workflows
* Cross-lingual transfer learning
* NLP evaluation techniques
* Error analysis methodologies
* Research-oriented machine learning experimentation

---

## Conclusion

This project presents a complete multilingual transformer NLP pipeline using DistilBERT for sentiment analysis. It combines practical transformer fine-tuning with research-oriented evaluation and multilingual experimentation, demonstrating both engineering implementation and analytical NLP understanding.

The work highlights the strengths and limitations of multilingual transformers in real-world sentiment classification tasks and provides a strong foundation for advanced multilingual NLP research.

---

## License

This project is intended for educational and research purposes.
