

# Chatbot Project

A simple Python chatbot demonstrating:

- **Scikit-learn** for sentiment classification  
- **Word2Vec** embeddings for feature extraction  
- **Stylistic analysis** (type-token ratio, punctuation density, etc.)  
- **Transcript processing** to separate chatbot and user utterances

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Getting the Word2Vec Model](#getting-the-word2vec-model)  
- [Customization](#customization)  
- [Transcript Processing](#transcript-processing)  
- [Advanced Transcript Analysis (Optional)](#advanced-transcript-analysis-optional)  
- [Tips & Troubleshooting](#tips--troubleshooting)  
- [License](#license)  

---

## Project Structure

```
.
├── analyze_transcripts.py     # Optional script for advanced transcript analysis (requires TAACO)
├── chatbot.py                 # Main chatbot code (dialogue states, sentiment classifier, etc.)
├── dataset.csv                # Training data for sentiment classification (sample or small data recommended)
├── process_transcripts.py     # Script to separate chatbot/user utterances into separate files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

> **Note**: A file named `w2v.pkl` (containing the pretrained Word2Vec model) is **not** included here because it is very large. See [Getting the Word2Vec Model](#getting-the-word2vec-model) below.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
   ```

2. **(Optional) Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   # or
   venv\Scripts\activate       # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) NLTK Data**

   - If you haven’t installed NLTK’s tokenizers before, in a Python shell:
     ```python
     import nltk
     nltk.download('punkt')
     ```
   - Or let the code handle the download automatically.

5. **Dataset**

   - If `dataset.csv` is large or private, you may need to provide your own smaller sample or update references in `chatbot.py`.

---

## Usage

### 1. Run the Chatbot

```bash
python chatbot.py
```

- The chatbot will ask for your name and proceed with a conversation.  
- It logs everything to a timestamped `.txt` file in the project directory.

### 2. Chatbot Flow

- **Sentiment Analysis**  
  Uses a scikit-learn model (e.g., SVM) trained on Word2Vec embeddings or TFIDF features.
- **Stylistic Analysis**  
  Reports metrics like average sentence length, punctuation density, type-token ratio, and more.

---

## Getting the Word2Vec Model

The Word2Vec model file, typically `w2v.pkl`, is **not** included in this repo because of its large size. You have a few options:

1. **Obtain the Pretrained Google News Word2Vec**  
   - Download from [this archive](https://code.google.com/archive/p/word2vec/), then convert it to `.pkl` format if needed.
   - Place `w2v.pkl` in the same directory as `chatbot.py`.

2. **Use Gensim’s Pretrained Models**  
   - You can load Google News Word2Vec via Gensim and save it locally:
     ```python
     import gensim.downloader as api

     # Download the Google News vectors (about 1.6GB)
     model = api.load("word2vec-google-news-300")

     # Save as a .pkl for use in chatbot.py
     model.save("w2v.pkl")
     ```
   - Again, put `w2v.pkl` in the project directory.

3. **Use Your Own Word2Vec**  
   - If you have a different pretrained model, rename or update `chatbot.py` references accordingly.

---

## Customization

- **Classifier Choice**  
  - In `chatbot.py`, check `instantiate_models()` to pick Naive Bayes, Logistic Regression, SVM, or MLP.
  - Adjust the calls to `train_model_tfidf(...)` or `train_model_w2v(...)` depending on your desired approach.

- **Stylistic Features**  
  - See `custom_feature_1()` and `custom_feature_2()` in `chatbot.py`. You can expand or modify them.

- **Dependency Parsing**  
  - If you don’t need dependency parses from Stanford CoreNLP, you can comment out the relevant lines (like `get_dependency_parse()`).

---

## Transcript Processing

If you have a transcript file like `test.txt` and want to separate chatbot and user utterances:

```bash
python process_transcripts.py
```

It generates:

- `all_test.txt`  
- `chatbot_test.txt`  
- `user_test.txt`

---

## Advanced Transcript Analysis (Optional)

If you install [**TAACO**](https://www.linguisticanalysistools.org/taaco.html) (a separate download), you can run:

```bash
python analyze_transcripts.py
```

- By default, this looks for a directory of processed transcripts (e.g., `processed_transcripts/`) and saves a CSV file with various coherence metrics.

---

## Tips & Troubleshooting

- **.gitignore**  
  - Recommend ignoring `venv/`, `.idea/`, `__pycache__/`, `*.pyc`, and any large data/model files like `w2v.pkl`.

- **Large Files**  
  - If you have a big dataset or model, consider external hosting or Gensim’s built-in downloads.

- **Environment Issues**  
  - Make sure your Python version matches the code (e.g., 3.8+).  
  - Install correct versions of scikit-learn, nltk, etc., as listed in `requirements.txt`.

---

## License

This project is shared for **educational and demonstration purposes**.

---
