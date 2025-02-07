# Malagasy Root Extraction (Seq2Seq Model)
> **#Malagasy #RootExtraction #NLP #Seq2Seq #LSTM #MachineLearning #HuggingFace #motmalgache #Tenymalagasy #Morphology #OpenSource #MITLicense #TensorFlow #Keras #Scraping #Dictionary #DataScience #LanguageProcessing**

This repository provides a sequence-to-sequence (Seq2Seq) model for mapping **inflected/derived Malagasy words** to their **root forms**, using data scraped from [Tenymalagasy / motmalgache.org](https://motmalgache.org/).

## Highlights
- **Data**: `dictionary.db` (uploaded to [Hugging Face](https://huggingface.co/torquenada/malagasy-seq2seq-stemmer)) containing `(derivative, root)` pairs.  
- **Architecture**: Character-level Seq2Seq with LSTMs (TensorFlow/Keras).  
- **Performance**:
  - ~99% accuracy on the test set
  - ~0.928 exact-match F1
  - ~0.912 character-level F1

## Table of Contents
- [Overview](#overview)
- [Setup and Requirements](#setup-and-requirements)
- [Dictionary Database (`dictionary.db`)](#dictionary-database-dictionarydb)
  - [Optional: Recreate `dictionary.db`](#optional-recreate-dictionarydb)
- [Training the Seq2Seq Model](#training-the-seq2seq-model)
- [Evaluation Results](#evaluation-results)
- [Inference Example](#inference-example)
- [Using on Hugging Face](#using-on-hugging-face)
- [License](#license)
- [Questions or Feedback?](#questions-or-feedback)

## Overview
Malagasy exhibits rich morphological patterns, resulting in many derived word forms from a single root. This project trains a **character-level Seq2Seq model** (with an Encoder-Decoder LSTM architecture in TensorFlow/Keras) to reverse these derivations—given an inflected word, predict its root.

## Setup and Requirements
- **Python >= 3.8**
- **Libraries**:
  - `requests`, `beautifulsoup4`, `tqdm` (if you want to run the scraper)
  - `numpy`, `matplotlib`, `scikit-learn`, `tensorflow (2.x)`, `sqlite3` (builtin in Python)
- **GPU** (optional): will speed up training significantly.

## Dictionary Database (`dictionary.db`)
I provide `dictionary.db` directly in the [Hugging Face repository](https://huggingface.co/torquenada/malagasy-seq2seq-stemmer). This file contains:
- A `root_words` table listing root words and their metadata.
- A `derivatives` table listing derived forms for each root.

If you only want to train or run inference using the data, simply download the `dictionary.db` from Hugging Face and place it in your local directory.

### Optional: Recreate `dictionary.db`
If you prefer to scrape the data yourself (e.g., for an updated dataset), you can use the provided Python script (`tanymalagasy.py`). The script will connect to **motmalgache.org**, scrape root words and their derivatives, and populate `dictionary.db`.

## Training the Seq2Seq Model
The main training script (e.g., `train_seq2seq.py`) uses `dictionary.db`. It performs:

1. **Data Retrieval**: Retrieves `(derivative, root)` pairs from `dictionary.db`.  
2. **Vocabulary Building**: Creates a character set including `<start>`, `<end>`, `<pad>`.  
3. **Train/Test Split**: Splits data (80% train, 20% test).  
4. **Seq2Seq Construction**:
    - Embedding layer  
    - LSTM encoder  
    - LSTM decoder  
    - Dense (softmax) for character predictions  
5. **Training & Validation**  
6. **Evaluation**:
    - Keras accuracy & loss  
    - Exact-match F1  
    - Character-level (micro) F1  

**Example Run**:
```bash
python train_seq2seq.py
```

Key Arguments (inside script):
- `epochs`: default 20  
- `batch_size`: default 64  
- `latent_dim`: 512  

## Evaluation Results
From a typical run using ~44,000 `(inflected, root)` pairs:

- **Test Accuracy**: 0.9919  
- **Exact-match F1**: 0.9280  
- **Character-level F1 (micro)**: 0.9125  

Loss & Accuracy Curves can be found in `loss_accuracy.png`.

## Inference Example
The script defines a function `stem_word(input_word)` to predict a root for a given Malagasy derivative:

```python
sample_word = "abiliana"
predicted_root = stem_word(sample_word)
print(f"Word: {sample_word}, Predicted root: {predicted_root}")
# Example output: Word: abiliana, Predicted root: bily
```

## Using on Hugging Face
We have uploaded:

- **Model File**: `seq2seq_stemmer.h5`  
  [Hugging Face: torquenada/malagasy-seq2seq-stemmer](https://huggingface.co/torquenada/malagasy-seq2seq-stemmer)

- **Dictionary**: `dictionary.db`  
  [Hugging Face: torquenada/dictionnaire-racines-malgache](https://huggingface.co/datasets/torquenada/dictionnaire-racines-malgache/blob/main/README.md)

You can directly load these from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/<username>/<repo_name>
cd <repo_name>
```

Use the model in your code:
```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("seq2seq_stemmer.h5")

# (Optional) Access the dictionary.db if needed
# e.g., to re-run or refine training with new data

# For inference, use a function like stem_word() defined
# in your training or utility script.
```

## Questions or Feedback?
Feel free to open an **issue** or **pull request** on GitHub, or contact us via **Hugging Face Discussions**.

**Enjoy exploring Malagasy morphology!**
