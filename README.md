# Natural Language Processing (NLP) — Complete Learning Guide

## 1. Introduction to NLP

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence (AI) that enables computers to understand, process, analyze, and generate human language.

NLP combines computer science, linguistics, mathematics, statistics, and machine learning to work with text and speech.

Examples of NLP applications include:
- Chatbots and virtual assistants
- Language translation
- Sentiment analysis
- Text classification
- Spam detection
- Text summarization
- Speech recognition
- Question answering
- Information extraction
- Search engines

### Example

Input:

`I love learning Natural Language Processing.`

An NLP system can process the sentence, identify its words, convert them into numerical representations, and use a model to perform a task such as sentiment analysis.

Possible output:

`Sentiment: Positive`

## 2. Important NLP Terminology

### 2.1 Corpus

A **corpus** is a collection of text documents used for linguistic analysis or machine learning.

Example: A collection of 10,000 movie reviews is a corpus.

### 2.2 Document

A document is one individual text item within a corpus.

Example: One customer review or one news article.

### 2.3 Vocabulary

Vocabulary is the set of unique tokens identified by a tokenizer or text-processing system.

Example:

`["I", "love", "NLP", "learning"]`

### 2.4 Token

A token is a unit of text, such as a word, subword, or punctuation mark.

Example:

`Natural Language Processing`

Possible tokens:

`["Natural", "Language", "Processing"]`

### 2.5 Syntax

Syntax refers to the rules governing how words are arranged to form phrases and sentences.

### 2.6 Semantics

Semantics refers to the meaning of words, phrases, and sentences.

### 2.7 Morphology

Morphology studies how words are formed from smaller meaningful units.

Example:

`unhappiness = un + happy + ness`

### 2.8 Phonetics

Phonetics studies speech sounds and how they are produced, transmitted, and perceived.

### 2.9 Pragmatics

Pragmatics studies how context influences the meaning of language.

Example: The sentence "Can you open the window?" is grammatically a question but is often used as a request.

## 3. Applications of NLP

| Application | Description | Example |
|---|---|---|
| Sentiment Analysis | Identifies opinions or emotional polarity | Positive or negative reviews |
| Text Classification | Assigns categories to text | Spam detection |
| Machine Translation | Translates text between languages | Hindi to English |
| Text Summarization | Produces a shorter version of text | Summarizing a report |
| Named Entity Recognition | Identifies entities in text | Person, organization, location |
| Information Extraction | Extracts structured facts | Extracting names and dates |
| Topic Modelling | Discovers themes in documents | Finding themes in news |
| Information Retrieval | Finds relevant documents | Search engines |
| Spell Checking | Detects spelling mistakes | Suggesting corrections |
| Question Answering | Answers questions from information | Document-based QA |
| Conversational AI | Supports human-computer dialogue | Customer support chatbot |

## 4. NLP Workflow

A typical NLP project follows these steps:

1. Collect text data.
2. Explore and understand the dataset.
3. Clean and preprocess the text.
4. Tokenize the text.
5. Convert text into numerical features or token IDs.
6. Split data into training, validation, and test sets.
7. Train a model.
8. Evaluate the model.
9. Improve the model.
10. Deploy and monitor the application.

The exact workflow depends on the task. For pretrained Transformer models, aggressive cleaning, stemming, and stop-word removal are often unnecessary.

**Important:** Fit learned preprocessing components, such as a TF-IDF vectorizer, using training data only. This prevents information from the test set leaking into training.

## 5. Python Libraries for NLP

### NLTK

Used for tokenization, stop words, stemming, lemmatization, and educational NLP experiments.

### spaCy

Used for efficient text processing, part-of-speech tagging, and named entity recognition.

### TextBlob

Provides simple interfaces for sentiment analysis and other text-processing tasks.

### scikit-learn

Provides TF-IDF, machine-learning algorithms, clustering, pipelines, and evaluation metrics.

### Gensim

Used for Word2Vec, document embeddings, and topic modelling.

### TensorFlow and Keras

Used to build and train deep-learning models, including RNNs, LSTMs, and GRUs.

### PyTorch

Used for implementing neural networks and deep-learning models.

### Hugging Face Transformers

Used to load pretrained Transformer models, tokenizers, and training tools.

### NumPy and pandas

Used for numerical computation and dataset manipulation.

### Installation

Install commonly used libraries:

```bash
python -m pip install numpy pandas matplotlib scikit-learn nltk spacy gensim textblob
```

For deep learning:

```bash
python -m pip install tensorflow torch transformers datasets
```

Install only the libraries needed by your project, and verify compatibility with your Python version.

## 6. Text Cleaning and Preprocessing

Text preprocessing transforms raw text into a format suitable for a particular NLP task.

Common operations include:
- Lowercasing
- Removing unnecessary HTML tags
- Handling URLs
- Normalizing whitespace
- Tokenization
- Handling punctuation
- Stop-word filtering
- Stemming
- Lemmatization
- Handling numbers and special characters

### 6.1 Lowercasing

Lowercasing converts uppercase letters to lowercase.

```python
text = "Natural Language Processing"
print(text.lower())
```

Output:

```text
natural language processing
```

Lowercasing can reduce vocabulary size, but preserve case when it carries useful information, such as the difference between `US` and `us`.

### 6.2 Removing HTML Tags

```python
import re

text = "<p>NLP is interesting.</p>"

clean_text = re.sub(r"<[^>]+>", " ", text)

print(clean_text.strip())
```

Output:

```text
NLP is interesting.
```

For complicated HTML, use an HTML parser instead of relying entirely on regular expressions.

### 6.3 Removing URLs

```python
import re

text = "Visit https://example.com to learn NLP."

clean_text = re.sub(r"https?://\S+|www\.\S+", "", text)

print(clean_text)
```

### 6.4 Removing Extra Spaces

```python
import re

text = "NLP    is     useful"

clean_text = re.sub(r"\s+", " ", text).strip()

print(clean_text)
```

Output:

```text
NLP is useful
```

### 6.5 Regular Expressions

Regular expressions, or regex, are patterns used to find, match, or replace text.

| Pattern | Meaning |
|---|---|
| `\d` | Matches one digit |
| `\d+` | Matches one or more digits |
| `\d*` | Matches zero or more digits |
| `\w+` | Matches one or more word characters |
| `\s+` | Matches one or more whitespace characters |
| `[A-Za-z]+` | Matches English alphabetic sequences |
| `[^a-zA-Z]` | Matches characters other than English letters |
| `https?://\S+` | Matches a simple URL pattern |

Example:

```python
import re

text = "NLP123 is useful in 2026."

result = re.sub(r"\d+", "", text)

print(result)
```

Output:

```text
NLP is useful in .
```

**Note:** Do not automatically remove all numbers or punctuation. They may be essential in medical text, dates, prices, scientific documents, and sentiment analysis.

## 7. Tokenization

Tokenization divides text into smaller units called tokens.

### 7.1 Word Tokenization

```python
from nltk.tokenize import word_tokenize

text = "NLP is interesting!"

tokens = word_tokenize(text)

print(tokens)
```

Typical output:

```python
['NLP', 'is', 'interesting', '!']
```

Depending on the NLTK version, tokenizer resources such as `punkt` and `punkt_tab` may need to be downloaded.

### 7.2 Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize

text = "NLP is useful. It helps computers process language."

sentences = sent_tokenize(text)

print(sentences)
```

### 7.3 Types of Tokenization

- **Sentence tokenization:** Splits paragraphs into sentences.
- **Word tokenization:** Splits text into word-like units.
- **Character tokenization:** Represents text as individual characters.
- **Subword tokenization:** Divides words into smaller units, commonly used by Transformer models.

For pretrained models, always use the tokenizer associated with the model.

## 8. Stop Words

Stop words are common words that may be removed in some NLP tasks.

Examples include:

`is`, `the`, `a`, `an`, `and`, `of`

Example using NLTK:

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

text = "This is a simple example of NLP."

stop_words = set(stopwords.words("english"))

tokens = word_tokenize(text.lower())

filtered_words = [
    word for word in tokens
    if word.isalpha() and word not in stop_words
]

print(filtered_words)
```

Stop-word removal is not appropriate for every task. Removing `not` from "not good" can reverse the intended sentiment.

## 9. Stemming and Lemmatization

Both techniques reduce variations of words, but they work differently.

### 9.1 Stemming

Stemming uses heuristic rules to remove word endings. The result may be a word fragment rather than a dictionary word.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["playing", "played", "plays", "studies"]

for word in words:
    print(word, "->", stemmer.stem(word))
```

### 9.2 Lemmatization

Lemmatization attempts to convert a word into its dictionary base form, called a lemma.

```python
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cars", pos="n"))
print(lemmatizer.lemmatize("running", pos="v"))
print(lemmatizer.lemmatize("better", pos="a"))
```

Possible output:

```text
car
run
good
```

| Stemming | Lemmatization |
|---|---|
| Uses heuristic rules | Uses linguistic resources |
| May produce word fragments | Aims to produce valid base forms |
| Often simpler and faster | May require more processing |
| Useful for rough normalization | Useful when linguistic accuracy matters |

Neither technique is mandatory for every NLP model. Modern Transformer models usually work with their own tokenization schemes.

## 10. Text Representation

Machine-learning models generally require numerical inputs. Text representation converts words and documents into numbers.

### 10.1 One-Hot Encoding

Each vocabulary word is assigned a unique position.

For the vocabulary:

`["cat", "dog", "bird"]`

Representations are:

```text
cat  -> [1, 0, 0]
dog  -> [0, 1, 0]
bird -> [0, 0, 1]
```

Limitation: one-hot vectors can be large and do not naturally represent semantic similarity.

### 10.2 Bag of Words

Bag of Words (BoW) represents a document using word occurrence counts. It does not preserve most word-order information.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I like NLP",
    "I like machine learning",
    "NLP helps learning"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 10.3 N-Grams

An n-gram is a sequence of `n` adjacent tokens.

- Unigram: `natural`, `language`
- Bigram: `natural language`
- Trigram: `natural language processing`

N-grams capture local word-order information but can increase the number of features significantly.

### 10.4 TF-IDF

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It gives greater importance to terms that are relatively distinctive within a document collection.

A simplified formula is:

TF-IDF(t, d) = TF(t, d) × IDF(t)

The precise formula depends on the implementation.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "NLP processes human language",
    "Language models learn patterns",
    "NLP models process text"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 10.5 Cosine Similarity

Cosine similarity measures the similarity between two nonzero vectors using the angle between them.

Cosine Similarity(A, B) = (A · B) / (||A|| × ||B||)

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(X)

print(similarity_matrix)
```

It is useful in document search, duplicate detection, and text similarity.

## 11. Word Embeddings

Word embeddings represent tokens as dense numerical vectors. They can capture useful relationships between words.

### 11.1 Word2Vec

Word2Vec learns word representations from surrounding-word patterns.

**CBOW (Continuous Bag of Words):** Predicts a target word from surrounding context words.

**Skip-gram:** Predicts surrounding context words from a target word.

### 11.2 FastText

FastText incorporates subword information, which can help represent words using character-level patterns.

### 11.3 GloVe

GloVe learns word vectors using global word co-occurrence statistics.

### 11.4 Doc2Vec

Doc2Vec learns vector representations for documents or larger text units.

### 11.5 Contextual Embeddings

Contextual models produce representations influenced by surrounding words.

For example, `bank` can have different meanings in:
- River bank
- Bank account

| Method | Main idea |
|---|---|
| Word2Vec CBOW | Context predicts target word |
| Word2Vec Skip-gram | Target predicts context |
| FastText | Uses subword information |
| GloVe | Learns from global co-occurrence |
| Doc2Vec | Represents documents |
| BERT embeddings | Context-sensitive representations |

Embeddings are used for similarity, clustering, retrieval, classification, and downstream neural networks.

## 12. Important NLP Tasks

### 12.1 Text Classification

Assigns a category to a text.

Examples:
- Spam or not spam
- Positive or negative review
- Sports, business, or technology article

### 12.2 Keyword Extraction

Identifies important terms in a document.

Methods include:
- TF-IDF
- Linguistic rules
- TextRank
- Learned extraction models

### 12.3 Topic Modelling

Discovers recurring themes in a collection of documents.

Latent Dirichlet Allocation (LDA) is a classical topic-modelling technique.

### 12.4 Information Retrieval

Finds and ranks documents relevant to a user's query. Methods include keyword matching, TF-IDF, dense-vector retrieval, and hybrid search.

### 12.5 Information Extraction

Extracts structured facts from unstructured text, such as names, dates, organizations, and relationships.

### 12.6 Machine Translation

Converts text from one language into another, such as Hindi to English.

### 12.7 Text Summarization

Produces a shorter version of a document while retaining important information.

### 12.8 Conversational AI

Conversational systems may use rules, intent classification, retrieval, language models, or combinations of these approaches.

A **closed-domain conversational agent** focuses on a restricted area, such as banking support, university admissions, or product assistance.

## 13. Part-of-Speech Tagging and Named Entity Recognition

### 13.1 Part-of-Speech Tagging

Part-of-speech (POS) tagging assigns grammatical labels to words.

Examples:
- Noun
- Verb
- Adjective
- Adverb
- Pronoun

### 13.2 Named Entity Recognition

Named Entity Recognition (NER) identifies entities such as:
- PERSON
- ORG — organization
- GPE — geopolitical entity
- DATE
- MONEY

Example using spaCy:

```python
import spacy

# Install the model once:
# python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. hired John Doe in London."

doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)

for entity in doc.ents:
    print(entity.text, entity.label_)
```

NER results depend on the model, context, language, and domain. A general model may need adaptation for specialized medical or legal text.

## 14. Sentiment Analysis

Sentiment analysis estimates the emotional tone or polarity expressed in text.

Common labels:
- Positive
- Negative
- Neutral

Example using TextBlob:

```python
from textblob import TextBlob

text = "The movie was excellent and enjoyable."

result = TextBlob(text).sentiment

print("Polarity:", result.polarity)
print("Subjectivity:", result.subjectivity)
```

- **Polarity:** Typically ranges from -1 (negative) to +1 (positive).
- **Subjectivity:** Typically ranges from 0 (more objective) to 1 (more subjective).

Simple sentiment tools may struggle with sarcasm, negation, slang, and domain-specific language.

## 15. Classical Machine Learning for NLP

Classical machine-learning algorithms remain useful for many text tasks.

Common algorithms:
- Naive Bayes
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- Random Forests
- Gradient-boosted models
- K-Means clustering

### Example: Text Classification

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I loved this film",
    "This film was wonderful",
    "I hated this film",
    "This film was terrible"
]

labels = [
    "positive",
    "positive",
    "negative",
    "negative"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)

prediction = model.predict(["The film was wonderful"])

print(prediction)
```

This is a small demonstration of the API, not a reliable evaluation. Real projects require adequate labelled data, proper train/test splitting, and appropriate metrics.

## 16. Deep Learning for NLP

Deep learning uses neural networks to learn useful patterns from text.

### 16.1 Embedding Layer

Maps token IDs to dense vectors that can be learned during training or initialized from pretrained embeddings.

### 16.2 Recurrent Neural Network (RNN)

Processes a sequence step by step while maintaining a hidden state.

RNNs are useful for learning sequential modelling, but standard RNNs may struggle with long-range dependencies.

### 16.3 Long Short-Term Memory (LSTM)

LSTM networks use gates and a cell state to help preserve information over longer sequences.

### 16.4 Gated Recurrent Unit (GRU)

GRUs use update and reset gates. They have a simpler architecture than LSTMs and can work well for sequence tasks.

### 16.5 Bidirectional RNN

Processes a sequence in forward and backward directions. It can use context from both sides when the complete input is available.

### 16.6 Sequence-to-Sequence Models

An encoder processes an input sequence, and a decoder generates an output sequence.

Applications include:
- Machine translation
- Text summarization
- Conversational systems

### 16.7 Teacher Forcing

During some sequence-to-sequence training procedures, the decoder receives the correct previous target token rather than its own previous prediction. During inference, the true target sequence is unavailable, so the model must generate tokens using its predictions.

## 17. Attention and Transformers

### 17.1 Attention

Attention allows a model to assign different importance to different input positions when constructing a representation.

### 17.2 Self-Attention

Self-attention relates tokens within the same sequence.

Scaled dot-product attention is expressed as:

Attention(Q, K, V) = softmax((QKᵀ) / √dₖ) V

Where:
- Q = Queries
- K = Keys
- V = Values
- dₖ = Dimension of the key vectors

### 17.3 Transformer Architecture

A Transformer commonly includes:
- Token embeddings
- Positional information
- Multi-head attention
- Feed-forward neural networks
- Residual connections
- Layer normalization

Encoder-decoder Transformers use both encoder and decoder components. Other architectures may use only an encoder or only a decoder.

### 17.4 Advantages of Transformers

- Efficient parallel processing during training
- Strong contextual representations
- Ability to model relationships between tokens
- Reuse of pretrained models for multiple tasks

Limitations include computational costs, context-window constraints, data requirements, and possible incorrect outputs.

## 18. BERT, GPT, and Transfer Learning

### 18.1 BERT

BERT is an encoder-based Transformer trained to learn bidirectional contextual representations.

Common applications:
- Text classification
- Named entity recognition
- Question answering
- Sentence representation

### 18.2 GPT

GPT-style models use decoder-only autoregressive Transformers that predict the next token based on preceding context.

Common applications:
- Text generation
- Summarization
- Question answering
- Conversational AI
- Code generation

### 18.3 Transfer Learning

Transfer learning reuses knowledge from a pretrained model for another task.

Typical workflow:
1. Select a suitable pretrained model.
2. Load its matching tokenizer.
3. Prepare task-specific data.
4. Fine-tune the model or use it directly.
5. Evaluate on held-out data.
6. Save the model and tokenizer.
7. Test and monitor deployment.

### 18.4 Fine-Tuning vs Feature Extraction vs Prompting

- **Feature extraction:** Use a pretrained model to create features while keeping its parameters fixed.
- **Fine-tuning:** Continue training some or all model parameters on task-specific data.
- **Prompting:** Give instructions or examples to a model without necessarily updating its weights.

The best choice depends on dataset size, cost, privacy, latency, and performance requirements.

## 19. Evaluation Metrics

Evaluation metrics measure how well an NLP model performs.

### 19.1 Accuracy

The proportion of predictions that are correct.

Accuracy = Correct Predictions / Total Predictions

### 19.2 Precision

Precision measures the proportion of predicted positive examples that are actually positive.

Precision = TP / (TP + FP)

### 19.3 Recall

Recall measures the proportion of actual positive examples correctly identified.

Recall = TP / (TP + FN)

### 19.4 F1-Score

F1-score is the harmonic mean of precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Here:
- TP = True Positives
- FP = False Positives
- FN = False Negatives

### 19.5 Confusion Matrix

A confusion matrix summarizes correct and incorrect predictions by class.

### 19.6 Metrics for Text Generation

- **BLEU:** An n-gram overlap metric often used in translation.
- **ROUGE:** Overlap-based metrics often used in summarization.
- **Perplexity:** Measures how well a language model predicts token sequences.

Automatic metrics do not fully measure factual accuracy, fluency, or usefulness. Human evaluation and task-specific testing may also be required.

## 20. Suggested NLP Repository Structure

```text
NLP/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_NLP_Introduction.ipynb
│   ├── 02_Tokenization_and_Preprocessing.ipynb
│   ├── 03_Stemming_and_Lemmatization.ipynb
│   ├── 04_BOW_TFIDF_and_Ngrams.ipynb
│   ├── 05_Word_Embeddings.ipynb
│   ├── 06_POS_and_NER.ipynb
│   ├── 07_Sentiment_Analysis.ipynb
│   ├── 08_Text_Classification.ipynb
│   ├── 09_RNN_LSTM_and_GRU.ipynb
│   ├── 10_Seq2Seq_and_Attention.ipynb
│   └── 11_Transformers_BERT_GPT.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
├── reports/
└── tests/
```

This is a suggested structure. Create folders when needed, and avoid committing credentials, private data, or large model files to a public repository.

## 21. NLP Learning Roadmap

### Stage 1: Foundations
- Understand NLP terminology and applications.
- Learn Python strings and regular expressions.
- Practise text cleaning and tokenization.
- Understand stop words, stemming, and lemmatization.

### Stage 2: Classical NLP
- Learn one-hot encoding and Bag of Words.
- Understand n-grams and TF-IDF.
- Calculate cosine similarity.
- Train a text classifier using scikit-learn.
- Learn accuracy, precision, recall, and F1-score.

### Stage 3: Linguistic Analysis and Embeddings
- Practise POS tagging and NER.
- Build sentiment analysis systems.
- Learn Word2Vec, CBOW, Skip-gram, FastText, GloVe, and Doc2Vec.
- Explore keyword extraction and topic modelling.

### Stage 4: Deep Learning
- Understand embeddings and sequence padding.
- Learn RNN, LSTM, and GRU.
- Implement sequence-to-sequence models.
- Understand attention.

### Stage 5: Modern NLP
- Learn self-attention and Transformers.
- Use pretrained tokenizers.
- Explore BERT and GPT-style models.
- Fine-tune a pretrained model.
- Evaluate and document model limitations.

### Suggested Projects
1. Spam message classifier
2. Product review sentiment analyser
3. News category classifier
4. Keyword extraction tool
5. FAQ search engine
6. Named entity extraction pipeline
7. Text summarization experiment
8. Transformer-based text classifier

For every project, document the problem, data source, preprocessing, baseline, evaluation metrics, limitations, and execution steps.

## 22. Common Problems and Good Practices

### Common Problems

- **Missing NLTK resources:** Download the resource required by the error message.
- **Missing spaCy model:** Install a compatible model, such as `en_core_web_sm`.
- **Vocabulary mismatch:** Use the same fitted vectorizer during training and inference.
- **Tokenizer mismatch:** Load the tokenizer associated with the pretrained model.
- **Over-cleaning:** Avoid removing meaningful punctuation, numbers, or negation without justification.
- **Data leakage:** Fit learned preprocessing only on training data.
- **Class imbalance:** Inspect class distributions and use suitable evaluation metrics.
- **Overfitting:** Compare training and validation results.
- **Reproducibility problems:** Record package versions, data sources, and model settings.

### Best Practices

- Begin with a simple baseline.
- Keep raw and processed data separate.
- Use reproducible preprocessing pipelines.
- Split data appropriately.
- Inspect model errors rather than relying on a single metric.
- Respect data licences, privacy, and consent.
- Never publish passwords, API keys, or private data.
- Save the model, tokenizer, and preprocessing configuration together.
- Test across relevant languages and user groups.
- Document limitations and avoid presenting predictions as guaranteed facts.

## 23. References and Learning Resources

- NLTK: https://www.nltk.org/
- spaCy: https://spacy.io/usage
- scikit-learn: https://scikit-learn.org/stable/user_guide.html
- Gensim: https://radimrehurek.com/gensim/
- TensorFlow Tutorials: https://www.tensorflow.org/tutorials
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- BERT Paper: https://arxiv.org/abs/1810.04805

## Conclusion

Natural Language Processing enables computers to work with human language. Learning NLP involves understanding text preprocessing, numerical text representations, classical machine learning, embeddings, deep learning, attention, and Transformer architectures.

**Recommended starting point:** Begin with tokenization and preprocessing, practise each example, then move to TF-IDF and text classification before studying embeddings and Transformers.
