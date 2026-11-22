# Recurrent Neural Network (RNN) - IMDB Sentiment Classification

A Keras model with a word-embedding layer and a `SimpleRNN` layer that classifies IMDB movie reviews as positive or negative.

## Files in this folder

| File | Description |
|---|---|
| [`RNN.ipynb`](RNN.ipynb) | Loads the pre-encoded IMDB reviews through Keras, pads them to 500 tokens, trains an Embedding + SimpleRNN + Dense model for 5 epochs, evaluates it on the test set and prints predictions for six test reviews. The notebook keeps the saved outputs of the author's run. |

## 1. Algorithm Name

Recurrent Neural Network (the Keras `SimpleRNN` layer, a fully connected recurrent layer) with a learned word embedding, used for binary text classification (sentiment analysis).

## 2. Definition

An RNN processes a sequence one element at a time and carries a hidden state from each step to the next. The hidden state acts as a running summary of everything read so far, so the same small set of weights can handle sequences of any length. For classification, the final hidden state is passed to an output layer.

## 3. How It Works

**General idea**

1. Each word is mapped to an integer index, and an **embedding layer** looks up a dense vector for that index.
2. The **recurrent layer** reads the vectors in order. At every step it combines the current word vector with the previous hidden state to produce a new hidden state.
3. After the last word, the final hidden state is fed to a **dense output layer**; for two classes a single sigmoid unit gives the probability of the positive class.
4. Training uses **backpropagation through time (BPTT)**: the network is unrolled over the sequence and gradients flow back through every step.

**What the notebook does**

1. Loads IMDB with `tf.keras.datasets.imdb.load_data(num_words=10000)`, keeping the 10,000 most frequent words.
2. Prints `x_train` (a NumPy array of integer lists).
3. Pads or truncates every review to 500 tokens with `sequence.pad_sequences(..., maxlen=500)`.
4. Builds the model below.
5. Compiles with `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`.
6. Trains with `epochs=5`, `batch_size=64`, `validation_split=0.2`.
7. Evaluates on the 25,000 test reviews and prints the test accuracy.
8. Predicts `x_test[5:11]` and prints the probability, predicted sentiment and an "Actual Sentiment" line for each (see the note in section 13).

**Model architecture** (output shapes and parameter counts are worked out from the layer definitions; the notebook does not call `model.summary()`)

| # | Layer as written in the code | Output shape (per review) | Trainable parameters |
|---|---|---|---|
| 1 | `layers.Embedding(max_features, 32)` with `max_features = 10000` | 500 x 32 | 320,000 |
| 2 | `layers.SimpleRNN(32)` (default `tanh`, returns only the last hidden state) | 32 | 2,080 |
| 3 | `layers.Dense(1, activation='sigmoid')` | 1 | 33 |
| | **Total** | | **322,113** |

## 4. Why It Is Used

- Text is ordered; an RNN reads it in order and can use earlier words when interpreting later ones, which a bag-of-words model cannot.
- The same weights are applied at every time step, so variable-length input is handled naturally.
- It is the simplest recurrent architecture and a common starting point before LSTM and GRU layers.

## 5. Mathematical Intuition

**Embedding lookup**: word index $w_t$ selects row $w_t$ of a trainable matrix $E \in \mathbb{R}^{10000 \times 32}$:

```math
x_t = E[w_t]
```

**Recurrent update** with $h_0 = 0$:

```math
h_t = \tanh\left(W_x x_t + W_h h_{t-1} + b\right)
```

Here $W_x$ is 32 x 32, $W_h$ is 32 x 32 and $b$ has 32 entries: $32 \cdot 32 + 32 \cdot 32 + 32 = 2080$ parameters.

**Output** after the last step $T = 500$:

```math
\hat{y} = \sigma\left(w^\top h_T + c\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
```

**Binary cross-entropy** for label $y \in \{0, 1\}$:

```math
L = -\left[\, y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \,\right]
```

**Vanishing gradients**: in BPTT the gradient that reaches step $t$ contains a product of Jacobians $\partial h_k / \partial h_{k-1} = \operatorname{diag}(1 - h_k^2)\, W_h$. Over hundreds of steps this product usually shrinks towards zero (or, less often, explodes), so a SimpleRNN struggles to learn from words far from the end of a 500-token sequence. LSTM and GRU layers were designed to reduce this problem.

## 6. Important Parameters

**General RNN parameters**

| Parameter | Meaning |
|---|---|
| Vocabulary size (`num_words`, `input_dim`) | Number of distinct word indices kept |
| Sequence length (`maxlen`) | Length that every sequence is padded or truncated to |
| Embedding dimension (`output_dim`) | Size of each word vector |
| `units` | Size of the hidden state |
| `activation` | Recurrent activation, `tanh` by default |
| `return_sequences` | Return every hidden state (needed for stacking RNN layers) or only the last |
| `dropout`, `recurrent_dropout` | Regularisation on inputs and on the recurrent connection |
| Optimizer, loss, `epochs`, `batch_size` | Training settings |

**Used in this code**

| Parameter | Value in `RNN.ipynb` |
|---|---|
| `max_features` (vocabulary size) | 10000 |
| `max_len` (sequence length) | 500 |
| Padding | `pad_sequences` defaults: zeros added at the start (`padding='pre'`); longer reviews keep their last 500 tokens (`truncating='pre'`) |
| Embedding dimension | 32 |
| SimpleRNN units | 32 |
| Output | `Dense(1, activation='sigmoid')` |
| Optimizer | `'adam'` (Keras default learning rate 0.001) |
| Loss | `'binary_crossentropy'` |
| Metric | `accuracy` |
| Epochs | 5 |
| Batch size | 64 |
| Validation | `validation_split=0.2` (Keras takes the last 20% of the training arrays: 20,000 reviews for training, 5,000 for validation) |
| Decision threshold | Probability > 0.5 is printed as Positive |

## 7. Advantages

- Handles sequences of any length with a fixed number of parameters.
- Uses word order, unlike bag-of-words models.
- Small and quick to build; a good baseline for sequence tasks.
- The same structure works for many-to-one (classification), one-to-many and many-to-many tasks.

## 8. Disadvantages

- Vanishing and exploding gradients make long-range dependencies hard to learn.
- Steps are processed one after another, so training parallelises poorly compared with convolutions or attention.
- Overfits small datasets quickly without regularisation.
- Usually outperformed by LSTM, GRU and Transformer models on real text tasks.

## 9. Real-World Applications

- Sentiment analysis of reviews and social-media posts
- Next-word prediction and simple language modelling
- Time-series forecasting (sensor readings, demand)
- Speech and handwriting recognition (historically, often with LSTM or GRU cells)
- Sequence labelling such as part-of-speech tagging

## 10. Python Libraries Used

| Library | Import in the notebook | Used for |
|---|---|---|
| TensorFlow / Keras | `import tensorflow as tf`, `from tensorflow.keras import datasets, layers, models` | Loading IMDB, building, training, evaluating and predicting |
| Keras preprocessing | `from tensorflow.keras.preprocessing import sequence` | `sequence.pad_sequences` |
| NumPy | `import numpy as np` | Imported; not used directly in the cells |

## 11. Dataset Information

| Item | Details |
|---|---|
| Name | IMDB movie-review sentiment dataset (Keras version) |
| Source | `tf.keras.datasets.imdb.load_data()` downloads `imdb.npz` (about 17 MB) on first use and caches it in `%USERPROFILE%\.keras\datasets` |
| Size | 50,000 reviews: 25,000 for training and 25,000 for testing, each split balanced between positive and negative |
| Labels | `1` = positive, `0` = negative |
| Encoding | Reviews are already lists of word indices ranked by frequency. Index `1` marks the start of a review and index `2` replaces words outside the 10,000-word vocabulary (both are visible in the saved `print(x_train)` output) |
| Shapes after padding | `x_train` (25000, 500), `x_test` (25000, 500) |

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../README.md):

```powershell
cd 02_Recurrent_Neural_Network
python -m jupyter notebook RNN.ipynb
```

Run the cells in order (Run > Run All Cells).

To execute the notebook without the browser and keep the author's saved outputs untouched:

```powershell
python -m jupyter nbconvert --to notebook --execute RNN.ipynb --output RNN_executed.ipynb
```

Notes:

- The first run needs internet access to download the IMDB data (about 17 MB).
- Training runs on the CPU on native Windows (TensorFlow 2.11 and later do not use the GPU there). In the saved log each epoch took about 21-24 seconds on the author's machine.

## 13. Expected Output

1. The printed `x_train` array: a NumPy array of Python lists of integers, shortened by NumPy with `...`.
2. A training log of 5 epochs with 313 steps each. Saved output from the author's run:

| Epoch | accuracy | loss | val_accuracy | val_loss |
|---|---|---|---|---|
| 1 | 0.7238 | 0.5282 | 0.8354 | 0.3837 |
| 2 | 0.8673 | 0.3331 | 0.8568 | 0.3570 |
| 3 | 0.9258 | 0.2023 | 0.8552 | 0.3788 |
| 4 | 0.9603 | 0.1224 | 0.8546 | 0.3991 |
| 5 | 0.9811 | 0.0637 | 0.8264 | 0.4764 |

In these numbers training accuracy keeps rising while validation loss is lowest after epoch 2 and rises afterwards, a typical sign of overfitting.

3. The test evaluation. Saved output from the author's run:

```text
782/782 ━━━━━━━━━━━━━━━━━━━━ 13s 17ms/step - accuracy: 0.8216 - loss: 0.4941

Test Accuracy: 0.8216000199317932
```

4. Six prediction blocks of the form:

```text
Review 1
Predicted Probability of Positive Review: <value between 0 and 1>
Predicted Sentiment: Positive | Negative
Actual Sentiment: Positive | Negative
```

Saved output from the author's run:

| Printed as | Predicted probability | Predicted sentiment | "Actual Sentiment" line |
|---|---|---|---|
| Review 1 | 0.10118565 | Negative | Negative |
| Review 2 | 0.99505144 | Positive | Positive |
| Review 3 | 0.06819697 | Negative | Positive |
| Review 4 | 0.9849787 | Positive | Negative |
| Review 5 | 0.9863495 | Positive | Positive |
| Review 6 | 0.9856678 | Positive | Positive |

Note on this cell: the predictions are made for `x_test[5:11]`, but the "Actual Sentiment" line reads `y_test[i]` for `i` = 0 to 5, which are the labels of `x_test[0:6]`. Each "Actual" line therefore belongs to a different review from the prediction above it, and the table cannot be read as correct or incorrect predictions. Reading `y_test[5 + i]` would align the two.

Numbers will differ from run to run because the weights are initialised randomly and no random seed is set.
