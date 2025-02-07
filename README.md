# Malagasy Root Extraction (Seq2Seq Model)

This repository provides a sequence-to-sequence (Seq2Seq) model for mapping inflected/derived Malagasy words to their root forms, using data I scraped from Tenymalagasy / motmalgache.org.

## Highlights

*   **Data:** `dictionary.db` (now uploaded to Hugging Face) containing (`derivative`, `root`) pairs.
*   **Architecture:** Character-level Seq2Seq with LSTMs.
*   **Performance:**
    *   ~99% accuracy on the test set
    *   ~0.928 exact-match F1
    *   ~0.912 character-level F1

## Table of Contents

*   Overview
*   Setup and Requirements
*   Dictionary Database (`dictionary.db`)
*   Training the Seq2Seq Model
*   Evaluation Results
*   Inference Example
*   Using on Hugging Face
*   License

## Overview

Malagasy exhibits rich morphological patterns, resulting in many derived word forms from a single root. This project trains a character-level Seq2Seq model (with an Encoder-Decoder LSTM architecture in TensorFlow/Keras) to reverse these derivations—given an inflected word, predict its root.

## Setup and Requirements

*   Python >= 3.8
*   Libraries:
    *   `requests`, `beautifulsoup4`, `tqdm` (if you want to run the scraper yourself)
    *   `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` (2.x), `sqlite3` (builtin in Python)
*   GPU is optional but will speed up training significantly.

## Dictionary Database (`dictionary.db`)

I provide `dictionary.db` directly in the Hugging Face repository. This file contains:

*   A `root_words` table listing root words and their metadata.
*   A `derivatives` table listing derived forms for each root.

If you only want to train or run inference using the data, simply download the `dictionary.db` from Hugging Face and place it in your local directory.

### (Optional) Recreate `dictionary.db`

If you prefer to scrape the data yourself (e.g., for an updated dataset), you can use the provided Python script (`tanymalagasy.py`). The script will connect to motmalgache.org, scrape root words and their derivatives, and populate `dictionary.db`.

## Training the Seq2Seq Model

The main training script (e.g., `train_seq2seq.py`) uses `dictionary.db`. It performs:

1.  Data Retrieval: Retrieves (`derivative`, `root`) pairs from `dictionary.db`.
2.  Vocabulary Building: Creates a character set including `<start>`, `<end>`, `<pad>`.
3.  Train/Test Split: Splits data (80% train, 20% test).
4.  Seq2Seq Construction:
    *   Embedding layer
    *   LSTM encoder
    *   LSTM decoder
    *   Dense (softmax) for character predictions
5.  Training & Validation
6.  Evaluation:
    *   Keras accuracy & loss
    *   Exact-match F1
    *   Character-level (micro) F1

**Example Run**

```bash
python train_seq2seq.py
