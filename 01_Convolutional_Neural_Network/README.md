# Convolutional Neural Network (CNN) - CIFAR-10 Image Classification

A Keras convolutional neural network that classifies the 32 x 32 colour images of the CIFAR-10 dataset into 10 classes.

## Files in this folder

| File | Description |
|---|---|
| [`CNN.ipynb`](CNN.ipynb) | Loads CIFAR-10 through Keras, scales pixels to 0-1, builds a three-convolution-layer CNN, trains it for 5 epochs, evaluates it on the test set and plots predictions for 10 test images. The notebook keeps the saved outputs of the author's run. |

## 1. Algorithm Name

Convolutional Neural Network (CNN) for multi-class image classification.

## 2. Definition

A CNN is a neural network that learns small filters (kernels) and slides them across an image. Each filter responds to a local pattern such as an edge, a colour blob or a texture. Stacks of convolution, non-linearity and pooling layers turn raw pixels into increasingly abstract feature maps, and fully connected layers map those features to one score per class.

## 3. How It Works

**General idea**

1. **Convolution** - each learnable kernel is slid over the input and produces a feature map that is high where its pattern appears.
2. **Activation** - ReLU keeps positive responses and sets negative ones to zero, which makes the network non-linear.
3. **Pooling** - max pooling keeps the strongest value in each small window. It shrinks the feature maps and makes the network less sensitive to small shifts.
4. **Stacking** - deeper convolution layers combine low-level features into higher-level ones (edges, then shapes, then object parts).
5. **Classification head** - the final feature maps are flattened and passed through dense layers that output one score (logit) per class.
6. **Training** - a softmax turns logits into probabilities, cross-entropy measures the error, backpropagation computes gradients, and an optimizer (Adam here) updates the weights.

**What the notebook does**

1. Loads CIFAR-10 with `datasets.cifar10.load_data()`.
2. Divides all pixel values by `255.0` so they lie between 0 and 1.
3. Plots the first 10 training images in a 2 x 5 grid with their class names.
4. Builds the model below with `models.Sequential`.
5. Compiles it with the `adam` optimizer, `SparseCategoricalCrossentropy(from_logits=True)` and the `accuracy` metric.
6. Trains for 5 epochs with `validation_data=(test_images, test_labels)`.
7. Evaluates on the test set and prints the test accuracy.
8. Wraps the model with a `Softmax` layer, predicts all test images and shows the first 10 with the predicted and actual class.

**Model architecture** (output shapes and parameter counts are worked out from the layer definitions; the notebook does not call `model.summary()`)

| # | Layer as written in the code | Output shape | Trainable parameters |
|---|---|---|---|
| 1 | `Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3))` | 30 x 30 x 32 | 896 |
| 2 | `MaxPooling2D((2,2))` | 15 x 15 x 32 | 0 |
| 3 | `Conv2D(64, (3,3), activation='relu')` | 13 x 13 x 64 | 18,496 |
| 4 | `MaxPooling2D((2,2))` | 6 x 6 x 64 | 0 |
| 5 | `Conv2D(64, (3,3), activation='relu')` | 4 x 4 x 64 | 36,928 |
| 6 | `Flatten()` | 1,024 | 0 |
| 7 | `Dense(64, activation='relu')` | 64 | 65,600 |
| 8 | `Dense(10)` (no activation, outputs logits) | 10 | 650 |
| | **Total** | | **122,570** |

## 4. Why It Is Used

- Images have strong local structure. Convolution exploits it: the same small kernel is reused at every position (weight sharing), so a CNN needs far fewer parameters than a fully connected network on the same pixels.
- Weight sharing makes feature detection work wherever the pattern appears in the image; pooling adds tolerance to small shifts.
- Features are learned from data instead of being designed by hand.

## 5. Mathematical Intuition

**Convolution** (implemented as cross-correlation in deep-learning libraries). For an input $X$ with $C$ channels, a $k \times k$ kernel $K_f$ and bias $b_f$ for filter $f$:

```math
Z_{i,j,f} = b_f + \sum_{c=1}^{C} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m,\,j+n,\,c}\; K_{f}(m, n, c)
```

**Output size** for input width $W$, kernel size $F$, padding $P$ and stride $S$:

```math
O = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1
```

With the Keras defaults used here (`padding='valid'`, so $P = 0$, and $S = 1$): $32 - 3 + 1 = 30$. A 2 x 2 max pool with stride 2 then gives $\lfloor 30 / 2 \rfloor = 15$.

**Parameters of a convolution layer**: $(k \cdot k \cdot C_{in} + 1) \cdot C_{out}$, for example $(3 \cdot 3 \cdot 3 + 1) \cdot 32 = 896$ for the first layer.

**ReLU**: $f(z) = \max(0, z)$.

**Softmax** turns the 10 logits $z$ into probabilities:

```math
p_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}
```

**Sparse categorical cross-entropy** for an integer label $y$: $L = -\log p_y$. With `from_logits=True` the loss applies the softmax internally, which is numerically more stable. That is why the last `Dense(10)` layer has no activation and why the notebook adds a separate `Softmax` layer only for prediction.

**Adam** keeps running averages of the gradient $g$ and its square and scales each step:

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\theta_t = \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
```

## 6. Important Parameters

**General CNN parameters**

| Parameter | Meaning |
|---|---|
| `filters` | Number of kernels in a convolution layer (number of output feature maps) |
| `kernel_size` | Height and width of each kernel |
| `strides`, `padding` | Step of the sliding window; `'valid'` (no padding) or `'same'` (keep size) |
| `activation` | Non-linearity, usually ReLU in hidden layers |
| `pool_size` | Window size of the pooling layer |
| Dense `units` | Width of the fully connected layers |
| Optimizer and learning rate | How weights are updated |
| Loss | Error to minimise; cross-entropy for classification |
| `epochs`, `batch_size` | Passes over the data and samples per gradient step |
| Regularisation | Dropout, weight decay, data augmentation (not used here) |

**Used in this code**

| Parameter | Value in `CNN.ipynb` |
|---|---|
| Input shape | `(32, 32, 3)` |
| Convolution filters | 32, 64, 64 |
| Kernel size | 3 x 3 (stride 1 and `padding='valid'` are Keras defaults) |
| Pooling | `MaxPooling2D((2,2))` after the first and second convolution |
| Dense layers | 64 units with ReLU, then 10 units without activation |
| Optimizer | `'adam'` (Keras default learning rate 0.001) |
| Loss | `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)` |
| Metric | `accuracy` |
| Epochs | 5 |
| Batch size | Not set, so the Keras default of 32 applies (1,563 steps per epoch in the saved log) |
| Validation data | The test set, `(test_images, test_labels)` |
| Pre-processing | Pixel values divided by `255.0` |
| Prediction | `tf.keras.Sequential([model, tf.keras.layers.Softmax()])`, then `np.argmax` |

## 7. Advantages

- Far fewer parameters than a fully connected network on raw pixels, thanks to weight sharing.
- Learns hierarchical visual features automatically.
- Robust to small translations of objects in the image.
- Convolutions run efficiently on GPUs and other accelerators.
- Trained convolutional layers can be reused for related tasks (transfer learning).

## 8. Disadvantages

- Needs a lot of labelled images to train well from scratch.
- Training is compute-heavy; large models need a GPU.
- Not inherently invariant to rotation or large changes in scale.
- Small networks like this one overfit without regularisation or augmentation.
- Learned features are hard to interpret.

## 9. Real-World Applications

- Medical image analysis (X-ray, CT and MRI screening)
- Object detection and lane detection for driver assistance
- Face verification and recognition
- Visual quality inspection in manufacturing
- Document and handwriting recognition (OCR)
- Satellite and aerial image classification

## 10. Python Libraries Used

| Library | Import in the notebook | Used for |
|---|---|---|
| NumPy | `import numpy as np` | `np.argmax` over predicted probabilities |
| Matplotlib | `import matplotlib.pyplot as plt` | Sample-image grid and prediction plots |
| TensorFlow / Keras | `import tensorflow as tf`, `from tensorflow.keras import datasets, models, layers` | Loading CIFAR-10, building, training and evaluating the model, softmax wrapper |

## 11. Dataset Information

| Item | Details |
|---|---|
| Name | CIFAR-10 |
| Source | `tf.keras.datasets.cifar10.load_data()` downloads the archive (about 170 MB) on first use and caches it in `%USERPROFILE%\.keras\datasets` |
| Size | 60,000 colour images of 32 x 32 pixels: 50,000 for training and 10,000 for testing, 6,000 per class |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (indices 0-9, the `class_names` list in the notebook) |
| Shapes | `train_images` (50000, 32, 32, 3), `train_labels` (50000, 1), `test_images` (10000, 32, 32, 3), `test_labels` (10000, 1) |
| Pre-processing | Division by `255.0` turns the `uint8` pixels into `float64` values between 0 and 1 (about 1.5 GB of memory for training and test images together) |

Note: the notebook passes the test set as `validation_data`, so the per-epoch `val_accuracy` and the final test accuracy are measured on the same 10,000 images. There is no separate validation split.

## 12. How to Execute

From the repository root, after following "Getting started" in the [root README](../README.md):

```powershell
cd 01_Convolutional_Neural_Network
python -m jupyter notebook CNN.ipynb
```

Run the cells in order (Run > Run All Cells).

To execute the notebook without the browser and keep the author's saved outputs untouched, write the result to a new file:

```powershell
python -m jupyter nbconvert --to notebook --execute CNN.ipynb --output CNN_executed.ipynb
```

Notes:

- The first run needs internet access to download CIFAR-10 (about 170 MB).
- Training runs on the CPU on native Windows: TensorFlow 2.11 and later do not use the GPU there (the saved output contains this warning). In the saved log each epoch took about 23-39 seconds on the author's machine.

## 13. Expected Output

1. A figure with the first 10 training images in a 2 x 5 grid, titled with their class names.
2. A Keras `UserWarning` recommending an `Input(shape)` object instead of `input_shape=` (present in the saved output; it does not stop the run).
3. A training log of 5 epochs with 1,563 steps each. Saved output from the author's run:

| Epoch | accuracy | loss | val_accuracy | val_loss |
|---|---|---|---|---|
| 1 | 0.4459 | 1.5181 | 0.5475 | 1.2292 |
| 2 | 0.5954 | 1.1372 | 0.6243 | 1.0705 |
| 3 | 0.6522 | 0.9846 | 0.6526 | 0.9822 |
| 4 | 0.6897 | 0.8840 | 0.6674 | 0.9524 |
| 5 | 0.7140 | 0.8125 | 0.6935 | 0.8797 |

4. The evaluation line and the test accuracy. Saved output from the author's run:

```text
313/313 - 3s - 9ms/step - accuracy: 0.6935 - loss: 0.8797

Test accuracy: 0.6934999823570251
```

5. Ten plots titled `Predicted: <class> | Actual: <class>` for test images 0-9. In the saved output from the author's run, 9 of the 10 predictions match the actual class; the fifth image (a frog) is predicted as `deer`.

Numbers will differ from run to run because the weights are initialised randomly and no random seed is set.
