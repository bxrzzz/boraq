---
title: "Module 6 — Neural Networks & Deep Learning (Olist Edition)"
subtitle: "From perceptrons to a neural classifier on Olist data — and a CNN side-quest on product photos"
author: "BePro AI/ML Mentorship Program"
date: "2026-05-04"
---

# Module 6 — Neural Networks & Deep Learning

| | |
|---|---|
| Module duration | 12 hours (6 classes × 2 h) |
| Weeks | 11–12 |
| Pre-requisite | Module 4 — `model_v1.pkl` baseline |
| Module deliverable | `model_v2_nn.pkl` — neural-net classifier on Olist tabular data + comparison with M4 RF baseline |
| Optional CV side-quest | Train a small CNN on Olist product photos (separate Kaggle download) |

---

## Module overview

Most students think "neural networks = deep learning = magic." The truth: a 1-layer NN with sigmoid activation **is** logistic regression. A 3-layer NN can model non-linear patterns the random forest from M4 already captured tree-style. The real value of neural nets in this course is unlocking *unstructured data* — images and text. We use Olist tabular data first (to compare honestly with M4) and then take a side-quest on product photos for CNN intuition.

By the end of M6, every student has trained a neural classifier on `olist_clean.parquet` and run a head-to-head comparison with the M4 random forest.

## Learning outcomes

| LO | Statement | Cognitive level |
|---|---|---|
| LO1 | Implement a single neuron / perceptron and explain how activation works | Apply |
| LO2 | Train a multi-layer feedforward network with backprop using `tensorflow.keras` | Apply |
| LO3 | Diagnose CNN architecture for image classification | Apply |
| LO4 | Recognise when to use RNN vs Transformer for sequential data | Understand |
| LO5 | Apply transfer learning to a pretrained model | Apply |
| LO6 | Compare a neural model honestly to a classical baseline on the same task | Evaluate |

## Class structure

| # | Title | Olist focus |
|---|---|---|
| 6-1 | Perceptrons & Activations | 1-neuron model = logistic regression on `is_late` (sanity check) |
| 6-2 | Backpropagation | 3-layer MLP on Olist `is_late` (the real M6 deliverable) |
| 6-3 | Convolutional Neural Networks | Side-quest: classify Olist product photos by category |
| 6-4 | Recurrent Neural Networks | Brief: review-text sequences (proper text work in M7) |
| 6-5 | Transfer Learning | Fine-tune a small pretrained model (Gold-only optional) |
| 6-6 | Lab — NN vs RF Bake-off | Ship `model_v2_nn.pkl` + comparison report |

---

# Class 6-1 — Perceptrons and Activation Functions

> **Today: a neural network is just stacked logistic regressions. We prove it on Olist.**

## Why this matters today

Demystifying. Once students see that a 1-layer NN with sigmoid = the M4 logistic regression with the same coefficients, the magic disappears. What's left is the *structure* — and structure is what makes deep learning work.

## Section 1 — The perceptron in plain English

A neuron takes inputs, multiplies them by weights, adds a bias, runs the sum through an activation function, outputs a number.

```
neuron_output = activation( w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b )
```

If activation = sigmoid, this **is** logistic regression. Same equation, different name.

## Section 2 — Activation functions

| Function | Output range | When |
|---|---|---|
| **Sigmoid** | (0, 1) | Final layer of binary classifier |
| **Softmax** | (0, 1), summing to 1 | Final layer of multi-class classifier |
| **ReLU** = max(0, x) | [0, ∞) | Hidden layers — the modern default |
| **Tanh** | (-1, 1) | Older default for hidden layers, still used in RNNs |
| **Linear** | (-∞, ∞) | Final layer of a regressor |

ReLU is the workhorse of every modern deep network. Memorise this list.

## Section 3 — Code: 1-neuron model on Olist

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_parquet('olist_clean.parquet').dropna(subset=['is_late'])
features = ['distance_km', 'log_freight', 'num_items', 'estimate_days', 'is_weekend']
X = df[features].values
y = df['is_late'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# One neuron, sigmoid output = logistic regression
model = Sequential([Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.1, verbose=2)

# Compare to scikit-learn LogisticRegression — should give very similar AUC
```

Tell students: **train the M4 logistic regression on the same scaled data and compare AUC.** They should be within 0.005. That's the moment the magic dissolves.

## Section 4 — Why we'll need more than 1 neuron

A single neuron draws a *linear* boundary. Olist `is_late` isn't truly linear — interactions like `distance × estimate` matter. To capture those, we add hidden layers (Class 6-2).

## Quick Check

1. A 1-neuron NN with sigmoid activation = which classical model?
2. What does ReLU return for input -3? For input +5?
3. Why is softmax used instead of sigmoid in multi-class problems?

## Today's deliverable

### Bronze
- Train the 1-neuron model. Compare AUC to your M4 logistic regression. Should match within 0.01.

### Gold
- Same. Plus: train it twice with different `random_state` for the train/test split. Confirm both runs give nearly identical AUC. Discuss in 1 paragraph why this stability matters for honest comparisons.

---

# Class 6-2 — Backpropagation

> **Today: the math behind learning. Then we train a real 3-layer net on Olist.**

## Why this matters today

Backprop is the algorithm that lets a neural network *learn*. Every framework (TensorFlow, PyTorch, JAX) implements it. We won't derive the math from scratch — we'll *use* it correctly. The 3-layer net we train today is the headline of M6.

## Section 1 — Forward pass

Data flows in through the input layer. Each layer multiplies by weights, applies activation, passes to the next. Output is a prediction.

## Section 2 — Loss

Compare prediction to truth. For binary classification: **binary cross-entropy**. For multi-class: **categorical cross-entropy**. For regression: **MSE**. The loss is one number — how wrong the network was on this batch.

## Section 3 — Backward pass

Backprop = applying the chain rule of calculus to compute the gradient of the loss with respect to *every* weight. Then we nudge each weight slightly in the direction that *decreases* loss.

```
weight_new = weight_old - learning_rate · gradient
```

That's gradient descent. Backprop is the efficient way to compute the gradient.

## Section 4 — Optimisers

| Optimiser | Behaviour |
|---|---|
| **SGD** | Plain gradient descent. Stable but slow. |
| **Adam** | Adaptive learning rates per weight. Fast convergence. **Default.** |
| **RMSprop** | Like Adam minus the momentum trick. Used in some RNN setups. |

Use Adam unless you have a reason not to.

## Section 5 — Code: 3-layer MLP on Olist

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC'],
)

# Class weights to handle the 7% / 93% imbalance
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
class_weight = {0: 1.0, 1: n_neg / n_pos}

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=256,
    class_weight=class_weight,
    verbose=2,
)
```

## Section 6 — Reading the training curve

Plot `history.history['loss']` vs `history.history['val_loss']`:

- Both decreasing → model is learning, not yet overfitting
- Train loss drops while val loss flattens → close to convergence
- Train loss drops while val loss rises → overfitting; stop training (early stopping)

```python
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(..., callbacks=[es])
```

## Section 7 — Comparing to M4 RF

```python
y_pred_proba = model.predict(X_test).flatten()
print(f"NN ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# vs your M4 RF
import joblib
rf_bundle = joblib.load('model_v1.pkl')
print(f"RF ROC-AUC: {roc_auc_score(y_test, rf_bundle['model'].predict_proba(X_test)[:,1]):.3f}")
```

Honest reality: **on Olist tabular data with ~15 features, a tuned random forest usually beats or ties a small MLP.** Neural nets shine on unstructured data (images, text, audio). On clean tabular data, gradient-boosted trees often win. **State this honestly in your report.** That's a sign of mature ML thinking.

## Quick Check

1. What does Adam do that plain SGD doesn't?
2. Train loss = 0.10, validation loss = 0.45. Diagnosis?
3. On 100k tabular rows with 15 features, would you expect an MLP or a tuned RF to win?

## Today's deliverable

### Bronze
- Train the 3-layer MLP above on Olist. Plot training curves. Print test AUC.

### Gold
- Tune the architecture: try 1, 2, 3, 4 hidden layers. Plot test AUC vs depth. Defend your final architecture in one paragraph.

---

# Class 6-3 — Convolutional Neural Networks

> **Today: the side-quest. Olist has product photos. Let's classify them by category.**

## Why this matters today

CNNs are the most-used deep architecture in industry. Without seeing one work on real images, students leave the program with only theoretical knowledge. The Olist product photos give us a real CV task tied to the same e-commerce theme.

## Section 1 — Convolution in plain English

Instead of treating an image as 224×224×3 = 150k independent pixels, a convolutional layer slides a small filter (e.g. 3×3) across the image, looking for **local patterns**. Early layers find edges, later layers find shapes, deepest layers find object parts.

## Section 2 — CNN building blocks

| Layer | What it does |
|---|---|
| **Conv2D** | Slides filters across the image |
| **MaxPool2D** | Downsamples (2×2 → 1×1, keeping the max) |
| **Dropout** | Randomly zeroes activations during training (regularisation) |
| **Flatten + Dense** | Final classifier layers |

A standard "small CNN" architecture:

```
Input (224, 224, 3)
  → Conv2D(32) → ReLU → MaxPool2D
  → Conv2D(64) → ReLU → MaxPool2D
  → Conv2D(128) → ReLU → MaxPool2D
  → Flatten
  → Dense(128) → ReLU → Dropout
  → Dense(num_classes) → Softmax
```

## Section 3 — On Olist product photos

The Olist product images are a **separate Kaggle download**: search "olist product images" or "brazilian e-commerce images". ~32k product photos, organised by `product_id`.

Pair with `olist_products_dataset.csv` to get categories. Filter to top 10 categories so you have enough data per class.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                horizontal_flip=True, rotation_range=10)
train_flow = train_gen.flow_from_directory(
    'olist_images_top10/',
    target_size=(96, 96),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)
val_flow = train_gen.flow_from_directory(
    'olist_images_top10/',
    target_size=(96, 96),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(96, 96, 3)),
    MaxPool2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(128, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=10)
```

Expected accuracy on top-10 categories: 50-70%. Random chance would be 10%. **State this honestly.** Olist photos are amateur quality and many categories overlap visually (kitchen vs houseware). Your model is learning real patterns even at 60%.

## Section 4 — Data augmentation

Real-world classifiers train on photos taken at slightly different angles, lighting, scales. Generate that diversity from your existing data:

```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
```

Free regularisation. Free generalisation. Use it on every CNN.

## Quick Check

1. Why use a 3×3 conv filter instead of feeding all 150k pixels into a Dense layer?
2. MaxPool downsamples 2×2 → 1×1. Why is downsampling helpful?
3. Why is data augmentation called "free regularisation"?

## Today's deliverable

### Bronze (optional — full credit without doing it)
- Watch the lecture, do the Quick Check, no code task. CV is hard to set up; skipping is acceptable.

### Gold (recommended)
- Train the small CNN on the Olist photos top-10 subset. Report test accuracy + confusion matrix. Identify which 2 categories the model confuses most often.

---

# Class 6-4 — Recurrent Neural Networks

> **Today: handling sequences. We preview review-text RNNs — the proper text work is in M7.**

## Why this matters today

Real-world data isn't always a fixed-shape vector. Olist review comments are sequences of words. Time-series data (web traffic, stock prices) is a sequence of numbers. RNNs handle sequences. We give a fast overview today; M7 does the deep work.

## Section 1 — What "sequential" means

A model that processes inputs **one at a time, carrying state between steps**. The state at step t depends on:
1. The current input
2. The state at step t-1

This lets the model remember context.

## Section 2 — The 3 sequence architectures

| Model | Strength | Weakness |
|---|---|---|
| **Vanilla RNN** | Simple | Can't remember long sequences (vanishing gradient) |
| **LSTM** | Long-term memory via gates | Slow, lots of parameters |
| **GRU** | Like LSTM, simpler & faster | Slightly less expressive |

For most real tasks now, you'd use a **Transformer** (M7), not an RNN. RNNs are still the right teaching ramp.

## Section 3 — Tiny LSTM on Olist review text (preview)

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM

reviews = pd.read_csv('olist_order_reviews_dataset.csv').dropna(subset=['review_comment_message'])
reviews['is_negative'] = (reviews['review_score'] <= 2).astype(int)

tok = Tokenizer(num_words=5000)
tok.fit_on_texts(reviews['review_comment_message'])
X = pad_sequences(tok.texts_to_sequences(reviews['review_comment_message']), maxlen=80)
y = reviews['is_negative'].values

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=80),
    LSTM(64),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.fit(X, y, validation_split=0.1, epochs=5, batch_size=128)
```

We're predicting "is this a negative review?" from the comment text. Expect AUC around 0.85 — decent for a 5-epoch tiny model.

## Section 4 — When NOT to use an RNN

- Tabular data — use trees/MLPs.
- Images — use CNNs.
- Long text or anything modern — use Transformers (M7).
- Short fixed-length sequences (e.g. 10 features) — use Dense layers.

## Quick Check

1. What's the vanishing gradient problem and which architecture solves it?
2. RNN, LSTM, GRU — which would you pick for a 200-token review classification today?
3. Why is Transformer the modern default over LSTM?

## Today's deliverable

### Bronze
- No code task. Read the lecture, answer Quick Check.

### Gold
- Train the tiny LSTM on Olist reviews. Report AUC. Compare to a TF-IDF + logistic regression baseline (you'll do this properly in M7 — preview it now).

---

# Class 6-5 — Transfer Learning

> **Today: stop training from scratch. Use a model someone else already trained on millions of images, fine-tune the last layer for your task.**

## Why this matters today

Training a CNN from scratch on Olist's 30k photos gives ~60% accuracy. Loading **MobileNetV2** pre-trained on ImageNet (1.2M photos, 1000 classes) and fine-tuning the last layer can lift that to 85%+ in 5 minutes of training. This is **the** technique that made deep learning useful for small companies.

## Section 1 — The intuition

A CNN trained on ImageNet has learned to detect edges, shapes, textures, parts. Those skills *transfer* to other image tasks. We:
1. Load the pre-trained model.
2. Freeze all its layers.
3. Replace the final classifier head with a fresh one for our task.
4. Train only the new head on our small dataset.

## Section 2 — Code

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base.trainable = False  # freeze

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=5)
```

5 epochs is often enough because the heavy lifting is already done.

## Section 3 — Fine-tuning (next-level move)

After the head has learned, **unfreeze the last few base layers** and continue training with a much lower learning rate:

```python
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)
model.fit(..., epochs=3)
```

This adapts ImageNet's last few layers specifically for Olist categories. Bumps accuracy by another 3-5%.

## Section 4 — When to use transfer learning

**Always for images, often for text.** Your dataset has fewer than ~100k labelled examples? Use transfer learning. Hugging Face ships a model for almost every common task.

## Quick Check

1. Why freeze base layers initially?
2. Why use a much smaller learning rate when fine-tuning?
3. ImageNet pre-trained weights are useful for *medical X-rays*. Why does that work?

## Today's deliverable

### Bronze
- No coding required (CV setup is heavy). Read + Quick Check.

### Gold (recommended for image-curious students)
- Re-run the M6-3 CNN task with MobileNetV2 transfer learning. Compare accuracy + training time to scratch CNN. Write a paragraph: when is transfer learning worth the setup cost?

---

# Class 6-6 — Lab: Neural Net vs Random Forest Bake-off

> **The 90-minute capstone. Train your best NN on Olist `is_late`. Compare to M4. Ship `model_v2_nn.pkl` + a comparison report.**

## Brief

Same task as M4 lab: predict `is_late` from `olist_clean.parquet`. This time use a neural network. Then run a fair head-to-head with your M4 random forest and report honestly which won.

## Lab structure (90 min)

### Stage 1 — Load + split (10 min)
- `olist_clean.parquet`. Same features and stratified split as M4 (`random_state=42` so it's apples-to-apples).
- Scale numeric features.

### Stage 2 — Build the NN (15 min)
- 3-layer MLP. Use class weights for imbalance.
- `Adam(1e-3)`, BCE loss, AUC metric.

### Stage 3 — Train with early stopping (15 min)
- 50 max epochs, `EarlyStopping(patience=5, restore_best_weights=True)`.
- Plot training + validation loss.

### Stage 4 — Evaluate (15 min)
- Test ROC-AUC, F1, classification report.
- Plot ROC curve.

### Stage 5 — Load M4 RF and compare (15 min)
```python
import joblib
rf_bundle = joblib.load('module-4/.../model_v1.pkl')
# Score both models on the SAME test set.
# Print side-by-side: AUC, F1 (late), recall, precision.
```

### Stage 6 — Save (5 min)
```python
joblib.dump({
    'model': model, 'scaler': scaler, 'features': features,
    'threshold': 0.4,
    'comparison': {'rf_auc': rf_auc, 'nn_auc': nn_auc}
}, 'model_v2_nn.pkl')
```

(For Keras models, also save `model.save('model_v2_nn.keras')`. The pickle then references the keras file.)

### Stage 7 — Honest report (15 min)

1-page markdown:
- AUC comparison table
- Which model won and by how much
- 2-3 reasons why (interpretation, not speculation)
- Which would you ship to production and why

## Submission

```
module-6/class_6/submissions/<TeamName>/
    ├── notebook.ipynb
    ├── model_v2_nn.keras
    ├── model_v2_nn.pkl
    └── comparison_report.md
```

## Grading rubric (100 points)

| Component | Weight |
|---|---|
| NN trains end-to-end | 15 |
| Honest comparison vs M4 RF (same test set) | 25 |
| Training curves plotted, overfitting diagnosed | 15 |
| Threshold + classification report on late class | 15 |
| Comparison report explains WHO won and WHY | 20 |
| Code + reproducibility | 10 |

> **Reminder:** the comparison + best model bundle feeds into M7 (where we add review-sentiment features and re-evaluate).
