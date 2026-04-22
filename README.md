# 🧠 Sentiment Analysis using RNN (TensorFlow / Keras)

## 📌 Project Overview

This project implements a **Sentiment Analysis model** using a **Recurrent Neural Network (RNN)** built with **TensorFlow/Keras**.

The model classifies input sentences into:

* ✅ **Positive Sentiment**
* ❌ **Negative Sentiment**

---

## 🚀 Features

* Text preprocessing using **Tokenizer**
* Sequence padding for uniform input size
* **Embedding layer** for word representation
* **SimpleRNN layer** for sequence learning
* Binary classification using **Sigmoid activation**
* Supports both:

  * Functional API (`Model`)
  * Sequential API (`Sequential`)
* Custom function for real-time sentiment prediction

---

## 🏗️ Model Architecture

```text
Input (Text)
   ↓
Tokenizer → Sequence → Padding
   ↓
Embedding Layer (Word → Vector)
   ↓
SimpleRNN Layer (Sequence Learning)
   ↓
Dense Layer (Sigmoid)
   ↓
Output (Positive / Negative)
```

---

## 📂 Dataset

The dataset consists of **40 sentences**:

* 20 Positive sentences → labeled as `1`
* 20 Negative sentences → labeled as `0`

Example:

```text
Positive: "I love this product"
Negative: "I hate this product"
```

---

## ⚙️ Preprocessing Steps

1. Convert text → numerical sequences using Tokenizer
2. Limit vocabulary using `vocab_size = 2000`
3. Handle unknown words using `<OOV>` token
4. Pad sequences to fixed length (`maxlen = 10`)

---

## 🧠 Key Concepts Used

### 🔹 Tokenization

Converts text into numbers:

```text
"I love AI" → [1, 2, 3]
```

### 🔹 Padding

Ensures equal input length:

```text
[1,2,3] → [1,2,3,0,0,...]
```

### 🔹 Embedding Layer

Transforms integers into dense vectors:

```text
Word Index → Vector Representation
```

### 🔹 RNN (SimpleRNN)

Processes sequences step-by-step and learns context.

---

## 🛠️ Model Implementation

### Functional API

```python
inp = Input(shape=(10,), dtype="int32")
x = Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
x = SimpleRNN(rnn_units)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=out)
```

### Sequential API

```python
model = Sequential([
    Input(shape=(10,)),
    Embedding(vocab_size, embed_dim, mask_zero=True),
    SimpleRNN(rnn_units),
    Dense(1, activation='sigmoid')
])
```

---

## ⚡ Training

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 🔍 Prediction Function

```python
def predict_sentiment(text):
    seq = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=10, padding='post')
    pred = model.predict(padded)[0][0]
    
    if pred > 0.5:
        return "Positive"
    else:
        return "Negative"
```

---

## 🧪 Example Usage

```text
Input: "I really love this!"
Output: Positive (Confidence: 0.87)

Input: "This is terrible"
Output: Negative (Confidence: 0.91)
```

---

## 📊 Important Parameters

| Parameter  | Value | Description          |
| ---------- | ----- | -------------------- |
| vocab_size | 2000  | Max words considered |
| maxlen     | 10    | Sequence length      |
| embed_dim  | 16    | Word vector size     |
| rnn_units  | 8     | RNN neurons          |

---

## 🧠 Learning Outcomes

This project helps understand:

* How text is converted into numerical data
* How RNN processes sequential data
* Difference between Functional and Sequential API
* Importance of embedding and padding

---

## 🚧 Limitations

* Small dataset (only 40 samples)
* Uses SimpleRNN (limited long-term memory)
* Not suitable for real-world large-scale NLP tasks

---

## 🚀 Future Improvements

* Replace SimpleRNN with **LSTM / GRU**
* Use larger datasets
* Add pre-trained embeddings (GloVe, Word2Vec)
* Improve accuracy with hyperparameter tuning

---

## 📦 Requirements

```bash
pip install tensorflow numpy pandas
```

---

## 🙌 Conclusion

This project demonstrates a **complete NLP pipeline**:

```text
Text → Preprocessing → Embedding → RNN → Prediction
```

It’s a great starting point for understanding **Deep Learning in NLP**.

---

## ⭐ If you found this useful

Give this repo a ⭐ and feel free to contribute!

