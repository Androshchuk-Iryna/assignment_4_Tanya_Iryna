# assignment_4_Tanya_Iryna

---

## Results Summary

| Model | Validation F1 | 
|-------|---------------|
| mBERT | 0.8512 |
| XLM-RoBERTa | 0.8698 | 
| mDeBERTa v3 | 0.8821 |


##  Data Insights

**Dataset:**
- Training samples: 1,010,000
- Test samples: 477
- Samples with locations: 23.1%
- Average text length: 14.6 words

**Examples:**
```
Text: "У Львові 34-річний мешканець Яворівського району..."
Locations: ['Львові', 'Яворівського району']

Text: "Нагадаємо, президент України Володимир Зеленський..."
Locations: ['України']
```

**Key Observations:**
- Class imbalance: 76.9% samples have no locations
- Short texts (good for RNN processing)
- Multiple locations per sample common
- High-quality Ukrainian annotations

---

## Metric Analysis
The competition uses **entity-level F1-score**, where each entity is a text span (start, end).
The model receives a score of 1 only when it has completely correctly restored the location boundaries.
Partial matches (for example, finding “Львів” instead of “місто Львів”) are counted as errors.

**Advantages:**

* A classic approach to NER, consistent with standards (CoNLL, spaCy).
* Clearly penalizes incorrect boundaries, which encourages high-quality sequence labeling.
* Stable and understandable metric for comparing models.

**Disadvantages:**

* Complete dependence on exact span boundaries: even a 1-character offset = 0 points.
* Does not take partial information into account - a model that “almost guessed right” gets the same score as a model that missed completely.
* Very sensitive to tokenization (Byte Pair Encoding can cut names into components -> F1 drop).

**Edge cases:**

* Complex toponyms such as “Яворівського району” may have different segmentations -> risk of incorrect boundaries.
* Locations with declension (Львів/Львові/Львова) - the model may consider them different tokens.
* Several locations in a sentence next to each other -> risk of confusing intervals.
* Phrases with quotation marks or punctuation (“в місті Києві,”) - often errors on commas.
---

## Validation Strategy

The task already provides for an official split via the is_valid field, where:

is_valid = 0 -> training data

is_valid = 1 -> validation data

This is not a random split - it was formed by the authors of the dataset in order to:

* maintain the same distribution of location types in train and validation;

* divide sentences from one document into different parts, which prevents data leakage;

* simulate the expected statistics on the test, which ensures correct correlation with the leaderboard.
---



## Data Preparation

### Tokenization & Label Alignment

```python
def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
        padding='max_length'
    )
    
    # Align labels with subword tokens
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100 if w is None else label[w] for w in word_ids]
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized
```

### Data Splits
- Training: 50,000 samples (is_valid=0)
- Validation: 10,000 samples (is_valid=1)
- Used official split to prevent data leakage

---

## Models & Architecture

### Transformer Models

All models use: **Transformer Encoder → Linear(hidden → 2) → Softmax(O/LOC)**

1. **mBERT** (`bert-base-multilingual-cased`)
   - 179M parameters, 104 languages

2. **XLM-RoBERTa** (`xlm-roberta-base`)
   - 278M parameters, improved cross-lingual transfer

3. **mDeBERTa v3** (`microsoft/mdeberta-v3-base`)
   - 278M parameters, disentangled attention


---

##  Training Configuration

```python
training_args = TrainingArguments(
    learning_rate=5e-5,              # Optimal after tuning
    per_device_train_batch_size=16,
    num_train_epochs=3,              # Prevents overfitting
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,                       # Mixed precision
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)
```

**Loss:** Cross-Entropy with ignore_index=-100 for padding tokens

---
