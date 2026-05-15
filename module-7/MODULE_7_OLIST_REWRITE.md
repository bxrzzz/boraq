---
title: "Module 7 — NLP & Computer Vision (Olist Edition)"
subtitle: "Mining Olist customer reviews and product photos"
author: "BePro AI/ML Mentorship Program"
date: "2026-05-04"
---

# Module 7 — NLP & Computer Vision

| | |
|---|---|
| Module duration | 12 hours (6 classes × 2 h) |
| Weeks | 13–14 |
| Pre-requisite | Module 6 — best classifier (`model_v1.pkl` or `model_v2_nn.pkl`) |
| Module deliverable | `model_v3_with_text.pkl` — Module 6 model enriched with review-sentiment feature, plus a lift comparison |

---

## Module overview

The Olist `review_comment_message` column contains 99k customer reviews in Portuguese. They're free text — useless to logistic regression *unless we convert them to numbers*. NLP is exactly that conversion.

By the end of M7, every student has:
1. Cleaned and tokenised real Portuguese review text.
2. Computed sentiment scores using a pre-trained multilingual transformer.
3. Joined those sentiment scores back to `olist_clean.parquet`.
4. Re-trained the M6 model with the new feature and measured the **lift**.

The CV side covers product photos — useful for portfolio breadth, optional for the compound deliverable.

## Learning outcomes

| LO | Statement | Cognitive level |
|---|---|---|
| LO1 | Preprocess raw text (tokenise, normalise, remove stopwords) | Apply |
| LO2 | Convert text to numerical features via TF-IDF and embeddings | Apply |
| LO3 | Use a pre-trained transformer model for sentiment classification | Apply |
| LO4 | Apply object detection / classification with a pre-trained CV model | Apply |
| LO5 | Use an LLM API to extract insight from unstructured text at scale | Apply |
| LO6 | Quantify the lift of adding a new feature to an existing model | Evaluate |

## Class structure

| # | Title | Olist focus |
|---|---|---|
| 7-1 | Text Preprocessing | Tokenise + normalise Olist Portuguese review comments |
| 7-2 | Word Embeddings | TF-IDF + Word2Vec on the reviews; baseline sentiment classifier |
| 7-3 | Transformers & Attention | Use `distilbert-base-multilingual-cased` for sentiment scoring |
| 7-4 | Computer Vision | Pre-trained ResNet for Olist product-category classification |
| 7-5 | Working with LLMs | Use an LLM API to summarise top complaints per category |
| 7-6 | Lab — Sentiment-Enriched Classifier | Add sentiment feature → train M6 model → measure lift |

---

# Class 7-1 — Text Preprocessing

> **Today: 99k Olist reviews. They're in Portuguese, full of typos, accents, emoji. We turn them into clean tokens.**

## Why this matters today

Models read numbers, not text. The first job is converting messy human writing into a normalised stream of tokens (word units). 80% of NLP code in production is preprocessing. Get this right and downstream embeddings/classifiers work. Get it wrong and you get garbage scores.

## Section 1 — Olist's review column

```python
reviews = pd.read_csv('olist_order_reviews_dataset.csv')
print(reviews.shape)                      # (99224, 7)
print(reviews['review_comment_message'].isna().sum())   # ~58k missing!
reviews = reviews.dropna(subset=['review_comment_message']).copy()
print(reviews.shape)                      # ~41k with text
```

**Reality check:** 58% of reviews have a 1-5 star score but no comment. Only 41k carry text. We work with the 41k.

## Section 2 — The classical preprocessing pipeline

```python
import re
import unidecode    # pip install unidecode

def clean_review(text):
    text = text.lower()
    text = unidecode.unidecode(text)        # remove accents: 'ção' → 'cao'
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep letters and spaces only
    text = re.sub(r'\s+', ' ', text).strip()
    return text

reviews['clean'] = reviews['review_comment_message'].apply(clean_review)
```

Decisions you need to write down for each project:

- **Lowercase?** Usually yes. "Excelente" vs "excelente" don't carry different meaning here.
- **Remove accents?** For Portuguese-trained classical models, yes. For multilingual transformers, NO — they're trained with accents.
- **Keep punctuation?** Usually no for TF-IDF. Yes for transformers (they tokenise it sensibly).
- **Remove emoji?** Depends. Sometimes "👍" carries sentiment signal — keep for transformers, drop for TF-IDF.

## Section 3 — Tokenisation

Tokenisation = splitting text into atoms (words, subwords, characters).

```python
# Word-level (simple)
tokens = reviews['clean'].apply(lambda t: t.split())

# Subword (modern — what transformers use). Don't roll your own; use the model's tokenizer.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
print(tok.tokenize("Adorei o produto, chegou rápido!"))
# ['ad', '##orei', 'o', 'produto', ',', 'chegou', 'rapido', '!']
```

For Class 7-2 (TF-IDF) → word-level. For Class 7-3 (transformer) → subword.

## Section 4 — Stopwords and stemming (classical only)

Stopwords = "the", "and", "of" — high-frequency, low-signal.

```python
from nltk.corpus import stopwords
import nltk; nltk.download('stopwords')
pt_stopwords = set(stopwords.words('portuguese'))

def remove_stopwords(tokens):
    return [t for t in tokens if t not in pt_stopwords]
```

For Olist reviews, removing stopwords before TF-IDF cleans signal nicely. **Do NOT remove stopwords before a transformer** — they need full context.

## Quick Check

1. 58% of Olist reviews have no comment. What's the implication for sample size?
2. When would you keep accents (`ção`) vs strip them?
3. Why use a model's own tokenizer for transformers instead of `text.split()`?

## Today's deliverable

### Bronze
- Load `olist_order_reviews_dataset.csv`. Drop NaNs. Apply `clean_review`. Print 5 examples before/after.

### Gold
- Same. Plus: build a Portuguese stopword list from your data (the 50 most-frequent tokens). Compare to NLTK's list. Justify any differences in 1 paragraph.

---

# Class 7-2 — Word Embeddings (and a baseline classifier)

> **Today: convert text to vectors. TF-IDF first, then Word2Vec. Train a baseline sentiment classifier.**

## Why this matters today

Once we have clean tokens we need to turn them into numbers. Two families:
1. **TF-IDF** — sparse, fast, interpretable. Strong baseline. Often beats fancy embeddings on classification with limited data.
2. **Word embeddings** (Word2Vec, GloVe, FastText) — dense vectors that capture semantic similarity (`good ≈ great`).

For Olist sentiment, TF-IDF + logistic regression gives a strong baseline. We'll beat it with a transformer in 7-3, but only after honest comparison.

## Section 1 — TF-IDF

**Term Frequency × Inverse Document Frequency.** A word's score is high when it appears often in *this* review and rarely in *other* reviews.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

reviews['is_negative'] = (reviews['review_score'] <= 2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    reviews['clean'], reviews['is_negative'],
    test_size=0.2, stratify=reviews['is_negative'], random_state=42
)

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_train_v = vec.fit_transform(X_train)
X_test_v = vec.transform(X_test)

clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train_v, y_train)
print(classification_report(y_test, clf.predict(X_test_v)))
```

Expect F1 around 0.78 for the negative class. **Solid baseline.**

## Section 2 — Inspecting what the model learned

```python
feature_names = vec.get_feature_names_out()
weights = clf.coef_[0]

top_neg_words = pd.Series(weights, index=feature_names).sort_values(ascending=False).head(20)
top_pos_words = pd.Series(weights, index=feature_names).sort_values().head(20)

print("Words pushing 'negative':", top_neg_words.index.tolist())
print("Words pushing 'positive':", top_pos_words.index.tolist())
```

You'll see negatives dominated by Portuguese: `nao recebi`, `pessimo`, `horrivel`, `defeito`. Positives: `otimo`, `excelente`, `super`.

This is the feature interpretability test — if the top words look sensible, the model is learning real signal.

## Section 3 — Word2Vec (briefly)

Word2Vec produces a dense vector per word, learned so that similar-context words have similar vectors.

```python
from gensim.models import Word2Vec

sentences = [r.split() for r in reviews['clean']]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=3, workers=4)

# Closest words to "produto"
print(w2v.wv.most_similar('produto', topn=5))
# Likely: 'mercadoria', 'item', 'pedido', ...
```

For sentence-level features we average word vectors:

```python
import numpy as np

def review_to_vec(text, model):
    tokens = [w for w in text.split() if w in model.wv]
    if not tokens:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in tokens], axis=0)

X_train_w2v = np.array([review_to_vec(t, w2v) for t in X_train])
```

## Section 4 — TF-IDF vs Word2Vec — when to use which

| | TF-IDF | Word2Vec |
|---|---|---|
| Speed | Fast | Fast (with pre-trained vectors) |
| Vocabulary scaling | Big sparse matrix | Fixed-size dense vector |
| Captures semantics | No | Yes |
| Beats on small datasets? | Often yes | Sometimes |

For Olist with 41k reviews, TF-IDF is often equal or better. Don't overthink — use TF-IDF as your baseline.

## Quick Check

1. Why does TF-IDF down-weight common words?
2. What does `ngram_range=(1, 2)` capture that `(1, 1)` misses?
3. Word2Vec gives you a vector per word. How do you get a vector per review?

## Today's deliverable

### Bronze
- Train the TF-IDF + logistic regression sentiment classifier. Print the top 10 negative and top 10 positive words.

### Gold
- Same. Plus train Word2Vec. Average word vectors → sentence vectors → train a logistic regression on those. Compare F1. Which won?

---

# Class 7-3 — Transformers and Attention

> **Today: load a pre-trained multilingual transformer, score every Olist review for sentiment, ship the resulting feature.**

## Why this matters today

Transformers are the current state of the art for almost every NLP task. The good news: you don't have to train one. Hugging Face has thousands of pre-trained models — including `distilbert-base-multilingual-cased`, which speaks Portuguese natively. We'll use it to score Olist reviews and produce a single sentiment number per row that we'll add to our M3 dataset in Class 7-6.

## Section 1 — Self-attention in plain English

A transformer reads all tokens in the input *at once*. For each token, it computes an "attention score" against every other token: how much should I pay attention to *this* word given *that* one? The result is a context-aware representation of every word.

That's the whole secret. Self-attention replaces RNN's sequential memory with parallel context.

## Section 2 — BERT, GPT, and the family tree

| Model | What it does best |
|---|---|
| **BERT (encoder)** | Understanding text — classification, NER, sentiment |
| **GPT (decoder)** | Generating text — chat, completion, summarisation |
| **T5 (encoder+decoder)** | Translation, summarisation |
| **DistilBERT** | Lighter / faster BERT for production |

For Olist sentiment we want **understanding** → DistilBERT.

## Section 3 — Code: score Olist reviews with multilingual DistilBERT

```python
from transformers import pipeline

# Multilingual sentiment classifier — pre-trained, ready to use
classifier = pipeline(
    'sentiment-analysis',
    model='nlptown/bert-base-multilingual-uncased-sentiment',
    device=-1,    # use CPU; if you have a GPU use device=0
)

# Test on one review
print(classifier("Adorei o produto, chegou rápido e a qualidade é ótima!"))
# [{'label': '5 stars', 'score': 0.93}]
```

This model outputs star ratings (1-5) — convenient because it directly maps to Olist's `review_score`.

## Section 4 — Score every Olist review (batched)

```python
from tqdm import tqdm

def to_score(label_dict):
    """Map '5 stars' → 5"""
    return int(label_dict['label'].split()[0])

reviews_with_text = reviews.dropna(subset=['review_comment_message']).copy()

predictions = []
batch = []
batch_size = 32

for text in tqdm(reviews_with_text['review_comment_message'].tolist()):
    batch.append(text[:512])    # truncate to model's max
    if len(batch) == batch_size:
        predictions.extend(classifier(batch))
        batch = []
if batch:
    predictions.extend(classifier(batch))

reviews_with_text['nlp_sentiment'] = [to_score(p) for p in predictions]
```

On 41k reviews CPU takes ~30 minutes. Acceptable for a one-shot batch job. Save the result so you don't re-run.

```python
reviews_with_text[['review_id', 'order_id', 'nlp_sentiment']].to_parquet('review_sentiment.parquet')
```

## Section 5 — Sanity check

```python
# Compare model output to the actual review_score
import seaborn as sns
sns.heatmap(
    pd.crosstab(reviews_with_text['review_score'], reviews_with_text['nlp_sentiment'], normalize='index'),
    annot=True, fmt='.2f', cmap='Blues'
)
```

You should see a strong diagonal — model agrees with users 60-70% of the time. Off-diagonal entries are interesting (long detailed reviews where model disagrees with the score the customer gave).

## Quick Check

1. Why is self-attention better than RNNs for long sequences?
2. BERT vs GPT — which would you pick for review sentiment classification, and why?
3. The model's output is "5 stars" but the user gave 3. Who's right? (Trick question — discuss.)

## Today's deliverable

### Bronze
- Run the pipeline on 200 random Olist reviews. Print 10 examples with text + predicted score + actual score.

### Gold
- Run on full 41k reviews. Save `review_sentiment.parquet`. Plot the confusion matrix `actual_score vs predicted_score`.

---

# Class 7-4 — Computer Vision Fundamentals

> **Today: pre-trained ResNet50. Classify Olist product photos by category in 30 minutes.**

## Why this matters today

Same logic as the transformers class — you don't train CV models from scratch in 2026. You take a pre-trained model and fine-tune. ResNet50, EfficientNet, ViT — all available, all instant.

## Section 1 — The CV task on Olist

Goal: given a product photo, predict its category (e.g. `bed_bath_table` vs `health_beauty`). Classic image classification.

Note: Olist photos are amateur quality, multi-product, sometimes blurry. Don't expect 95% accuracy. **80% on top-10 categories is excellent.**

## Section 2 — Loading ResNet50

```python
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
```

`include_top=False` — drop the 1000-class classifier head. We'll add our own head with the right number of Olist categories.

## Section 3 — Fine-tune for Olist categories

(Same architecture as M6-5 but with ResNet base.)

```python
model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_olist_categories, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=10)
```

Top-10 categories on Olist usually trains to **75-85% accuracy** in 10 epochs.

## Section 4 — Object detection (briefly)

YOLOv8 (`pip install ultralytics`) gives you bounding-box object detection in 3 lines:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('olist_product.jpg')
results[0].show()
```

Useful for product-photo QA: detect "is there a phone in this photo?" before approving the listing.

## Section 5 — When to use which CV approach

| Task | Model |
|---|---|
| Classify image into one of N classes | Fine-tuned CNN (ResNet, EfficientNet) |
| Detect + locate objects | YOLO |
| Segment pixel-by-pixel | U-Net, Mask R-CNN |
| Generate / edit images | Stable Diffusion (out of scope here) |

## Quick Check

1. ImageNet was trained on 1000 classes. Why does its weights help for the 10-category Olist task?
2. `include_top=False` — what does it drop?
3. Olist accuracy plateaus at 80%. Should you try a deeper model or get more data?

## Today's deliverable

### Bronze
- No code task. Watch / read.

### Gold
- Train the ResNet50 classifier on Olist photos top-10 subset. Report accuracy + confusion matrix. Identify the 2 most-confused category pairs.

---

# Class 7-5 — Working with LLMs

> **Today: when classification isn't enough — generate summaries.**

## Why this matters today

Sometimes the business question is open-ended. *"What are customers complaining about in the home_decor category?"* You can't answer that with a classifier. You need a model that reads the review and *summarises* it. That's what an LLM does.

## Section 1 — The API call

```python
from openai import OpenAI
client = OpenAI(api_key='...')

reviews_home = reviews_with_text[
    (reviews_with_text['nlp_sentiment'] <= 2)
    & (reviews_with_text['product_category_english'] == 'bed_bath_table')
]['review_comment_message'].sample(50).tolist()

prompt = f"""You are a product analyst. Read these 50 negative customer reviews about
bed/bath/table products on a Brazilian e-commerce site. Identify the 5 most common
complaint themes. Output a numbered list with one sentence per theme.

Reviews:
{chr(10).join(f'- {r}' for r in reviews_home)}
"""

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': prompt}],
)
print(response.choices[0].message.content)
```

Output is a plain-English summary the marketing team can read.

## Section 2 — Prompt engineering basics

| Pattern | Example |
|---|---|
| **Role assignment** | "You are a product analyst..." |
| **Specific instruction** | "Output a numbered list with one sentence each." |
| **Concrete output format** | "Each item: [theme]: [example review snippet]." |
| **Few-shot** | Provide 1-2 example outputs before asking for the answer. |

Vague prompts → vague answers. Specific prompts → specific answers.

## Section 3 — Costs and rate limits

`gpt-4o-mini` costs roughly $0.15/M input tokens. 50 reviews × 200 chars ≈ 4000 tokens ≈ $0.001 per call. Cheap for one-off analysis. **Costly if you call it on every order.** For per-order labelling, prefer the M7-3 transformer (free after download).

## Section 4 — When LLM, when transformer

| Task | Best tool |
|---|---|
| Classify each review's sentiment 1-5 (millions of rows) | M7-3 transformer (free, batched, fast) |
| Summarise top complaints in a category (one-shot) | LLM API |
| Translate Portuguese to English | Either; LLM gives more natural results |
| Extract structured fields (e.g. "what product does this review mention?") | LLM with structured output |

## Quick Check

1. Why use a transformer (not an LLM) for per-review sentiment scoring?
2. What's the cost difference between LLM and transformer at 100k reviews?
3. Name three prompt-engineering tactics from the table.

## Today's deliverable

### Bronze
- Pick one Olist category. Pull 30 negative reviews. Prompt an LLM (or Claude / Gemini) for top complaints. Paste the response.

### Gold
- Do this for 5 categories. Build a "Top complaints by category" report.

---

# Class 7-6 — Lab: Sentiment-Enriched Classifier (Module 7 capstone)

> **The 90-minute capstone. Add the M7-3 sentiment feature to your M6 model. Measure the lift. Ship `model_v3_with_text.pkl`.**

## Brief

Take `olist_clean.parquet` and `review_sentiment.parquet` (from M7-3). Join them. Re-train the M6 classifier with the new `nlp_sentiment` feature. Compare AUC to M6 baseline. Honest report.

## Lab structure (90 min)

### Stage 1 — Join data (15 min)
```python
clean = pd.read_parquet('olist_clean.parquet')
sentiment = pd.read_parquet('review_sentiment.parquet')

# An order can have multiple reviews — take mean
sent_per_order = sentiment.groupby('order_id')['nlp_sentiment'].mean().reset_index()
df = clean.merge(sent_per_order, on='order_id', how='left')

# Many orders won't have reviews. Decision: keep them but flag.
df['has_review'] = df['nlp_sentiment'].notna().astype(int)
df['nlp_sentiment'] = df['nlp_sentiment'].fillna(3)   # neutral default
```

### Stage 2 — Re-train M6 model with new feature (15 min)
- Same architecture as M6-6 (3-layer MLP) and M4-6 (RF). Add `nlp_sentiment` and `has_review` to the feature list.
- Same train/test split, same `random_state=42`. **Apples-to-apples comparison.**

### Stage 3 — Compare to baseline (15 min)
```python
print(f"M4 RF baseline (no sentiment): AUC = {auc_baseline:.3f}")
print(f"M6 NN baseline (no sentiment): AUC = {auc_nn:.3f}")
print(f"M7 NN with sentiment: AUC = {auc_v3:.3f}")
print(f"Lift: +{(auc_v3 - auc_nn)*100:.2f} percentage points AUC")
```

Honest reality: sentiment may give a **small lift (1-3 AUC points)** because:
- 58% of reviews have no comment → most rows get the default neutral sentiment.
- Late deliveries often *cause* negative reviews, not the other way — feature is partly downstream of the target.

State this honestly in your report. **Mature ML thinking acknowledges feature limitations.**

### Stage 4 — Permutation importance (10 min)
Which features really matter in the new model?

```python
from sklearn.inspection import permutation_importance
imp = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42)
ranking = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False)
print(ranking)
```

### Stage 5 — Save + report (15 min)
```python
joblib.dump({
    'model': best_model, 'scaler': scaler, 'features': features,
    'threshold': 0.4,
    'auc': auc_v3,
}, 'model_v3_with_text.pkl')
```

1-page report:
- AUC table (M4 / M6 / M7) — honest comparison
- Top 5 features by permutation importance
- Did sentiment help? By how much?
- 1 paragraph: why the lift was big / small / negative

### Stage 6 — Submission

```
module-7/class_6/submissions/<TeamName>/
    ├── notebook.ipynb
    ├── model_v3_with_text.pkl
    └── lift_report.md
```

## Grading rubric (100 points)

| Component | Weight |
|---|---|
| Sentiment data correctly joined to M3 spine | 15 |
| Same train/test split as M4/M6 (apples-to-apples) | 15 |
| Lift number reported, even if tiny | 25 |
| Permutation importance computed | 15 |
| Report is honest about limitations | 20 |
| Code + reproducibility | 10 |

> **Reminder:** `model_v3_with_text.pkl` is the model that gets deployed in M8.
