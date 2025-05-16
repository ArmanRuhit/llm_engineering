# Day 2: NLP Tasks with Transformers

## Table of Contents
1. [Setup and Dependencies](#setup-and-dependencies)
2. [Sentiment Analysis](#sentiment-analysis)
3. [Named Entity Recognition (NER)](#named-entity-recognition-ner)
4. [Question Answering](#question-answering)
5. [Text Summarization](#text-summarization)
6. [Text Generation](#text-generation)

## Setup and Dependencies

### Key Libraries
- `transformers`: For pre-trained NLP models
- `torch`: PyTorch for tensor operations
- `datasets`: For loading datasets

### Environment Setup
```python
import torch
from transformers import pipeline

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")
```

## Sentiment Analysis

### Model Initialization
```python
classifier = pipeline("sentiment-analysis", device=device)
```

#### Parameters:
- `task="sentiment-analysis"`: Specifies sentiment classification
- `device`: Automatically uses GPU if available

### Making Predictions
```python
result = classifier("I'm super excited to be on the way to LLM mastery")
print(result)
```

#### Output Format:
```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

#### Key Points:
- Returns sentiment label (POSITIVE/NEGATIVE) with confidence score
- Handles multiple texts if passed as a list
- Automatically downloads the default model if not cached

## Named Entity Recognition (NER)

### Model Initialization
```python
ner = pipeline("ner", device=device)
```

#### Parameters:
- `task="ner"`: Specifies named entity recognition
- `aggregation_strategy`: Optional, controls how overlapping entities are handled

### Extracting Entities
```python
result = ner("Barack Obama was the 44th president of the United States")
print(result)
```

#### Output Format:
```
[
  {'entity': 'I-PER', 'score': 0.999, 'word': 'Barack', 'start': 0, 'end': 6},
  {'entity': 'I-PER', 'score': 0.998, 'word': 'Obama', 'start': 7, 'end': 12},
  ...
]
```

#### Entity Types:
- `PER`: Person
- `ORG`: Organization
- `LOC`: Location
- `MISC`: Miscellaneous

## Question Answering

### Model Initialization
```python
question_answerer = pipeline("question-answering", device=device)
```

### Extracting Answers
```python
result = question_answerer(
    question="Who was the 44th president of the United States?",
    context="Barack Obama was the 44th president of the United States."
)
print(result)
```

#### Output Format:
```
{
  'score': 0.9889,
  'start': 0,
  'end': 12,
  'answer': 'Barack Obama'
}
```

#### Key Points:
- Extracts answer spans from provided context
- Returns confidence score and character positions
- Handles cases where the answer might not be in the context

## Text Summarization

### Model Initialization
```python
summarizer = pipeline("summarization", device=device)
```

### Generating Summaries
```python
text = """The Hugging Face transformers library is an incredibly versatile..."""
summary = summarizer(
    text,
    max_length=50,
    min_length=25,
    do_sample=False
)
print(summary[0]['summary_text'])
```

#### Parameters:
- `max_length`: Maximum length of the summary
- `min_length`: Minimum length of the summary
- `do_sample`: Whether to use sampling; set to `False` for greedy decoding

## Text Generation

### Model Initialization
```python
generator = pipeline("text-generation", model="gpt2", device=device)
```

### Generating Text
```python
generated = generator(
    "The future of AI is",
    max_length=50,
    num_return_sequences=2
)
```

#### Parameters:
- `max_length`: Maximum length of generated text
- `num_return_sequences`: Number of sequences to generate
- `temperature`: Controls randomness (lower = more deterministic)

## Best Practices

1. **Model Selection**:
   - Choose task-specific models for better performance
   - Consider model size vs. accuracy trade-offs

2. **Performance**:
   - Batch processing for multiple inputs
   - Use `torch.no_grad()` during inference to save memory

3. **Error Handling**:
   - Handle cases where input exceeds model's maximum sequence length
   - Implement rate limiting for API deployments

4. **Advanced Features**:
   - Explore model-specific parameters for fine-tuning
   - Consider distillation for deployment to resource-constrained environments
