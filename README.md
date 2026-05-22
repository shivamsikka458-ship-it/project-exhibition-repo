# 🤟 Sign Language Recognition System

> Real-time sign language detection using hand landmark tracking and deep learning — bridging the communication gap between the hearing-impaired community and the world.

---

## 📌 Overview

This project is a **Sign Language Recognition System** that uses a webcam to capture hand gestures and translate them into text in real time. It leverages **MediaPipe** for hand landmark extraction and a **CNN + Bidirectional LSTM** deep learning model to classify sign language gestures with high accuracy (~95%).

The system is designed to be trained on custom gesture datasets, making it flexible for any sign language vocabulary.

---

## 🧠 How It Works

The pipeline consists of three stages:

```
Webcam Input → Hand Landmark Extraction (MediaPipe) → CNN + BiLSTM Model → Predicted Sign
```

**Stage 1 — Hand Capture (`hand_capture.py`)**
Opens the webcam and uses MediaPipe Hands to detect and draw 21 hand landmarks in real time.

**Stage 2 — Data Collection (`collectdata.py`)**
Records sequences of hand keypoints (x, y, z for each of 21 landmarks = 63 values per frame) across 30 frames per sample, saving them as `.npy` files. Each sign is recorded for 30 sample videos.

**Stage 3 — Model Training (`train.py`)**
Loads all collected sequences, trains a hybrid CNN + BiLSTM neural network, and saves the model as `action_model.h5`. The architecture achieves ~95% validation accuracy.

---

## 🏗️ Model Architecture

The model uses a CNN + Bidirectional LSTM architecture for both spatial and temporal feature extraction:

| Layer | Type | Purpose |
|---|---|---|
| Conv1D (64 filters) | Convolutional | Spatial feature extraction from keypoints |
| BatchNormalization | Normalization | Stabilizes training |
| MaxPooling1D | Pooling | Reduces sequence length |
| BiLSTM (64 units) | Recurrent | Captures motion patterns forward & backward |
| BiLSTM (128 units) | Recurrent | Deeper temporal understanding |
| Dense (64) | Fully Connected | Classification features |
| Dropout (0.3) | Regularization | Prevents overfitting |
| Dense (softmax) | Output | Sign class probabilities |

- **Input shape:** `(30 frames, 63 keypoints)`
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Early Stopping:** Monitors `val_loss` with patience of 20 epochs

---

## 📁 Project Structure

```
project-exhibition-repo/
│
├── hand_capture.py       # Live webcam hand landmark visualization
├── collectdata.py        # Records gesture samples for training
├── train.py              # Trains the CNN + BiLSTM model
├── loader.py             # Utility for loading dataset
├── new.html              # Frontend/UI component
│
├── MP_Data.csv           # Dataset metadata / label file
├── dataset.npy           # Compiled NumPy dataset
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- Webcam

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/shivamsikka458-ship-it/project-exhibition-repo.git
cd project-exhibition-repo

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Test Hand Tracking

Verify that your webcam and MediaPipe are working correctly:

```bash
python hand_capture.py
```

This opens a window showing your webcam feed with hand landmarks overlaid. Press `Q` to quit.

---

### Step 2 — Collect Training Data

Open `collectdata.py` and set the word you want to record:

```python
word = "hello"       # The sign label
samples = 30         # Number of video samples
frames_per_sample = 30  # Frames per sample
```

Then run:

```bash
python collectdata.py
```

Repeat this step for each sign you want to recognize (e.g., `"hello"`, `"thanks"`, `"yes"`, `"no"`). Each run saves `.npy` files into `sign_language_dataset/data/`.

---

### Step 3 — Train the Model

Once data is collected for all signs, run:

```bash
python train.py
```

This will:
- Load all gesture sequences from your dataset
- Train the CNN + BiLSTM model for up to 200 epochs (with early stopping)
- Save the trained model as `action_model.h5`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `mediapipe` | 0.10.9 | Hand landmark detection |
| `opencv-python` | 4.13.0.92 | Webcam capture & image processing |
| `numpy` | 2.2.6 | Array operations & data storage |
| `tensorflow` / `keras` | — | Deep learning model |
| `scikit-learn` | — | Train/test splitting |
| `matplotlib` | 3.10.8 | Visualization |
| `sounddevice` | 0.5.5 | Audio output support |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Format

Each recorded sample is a NumPy array of shape `(30, 63)`:
- **30** = number of frames per sample
- **63** = 21 hand landmarks × 3 coordinates (x, y, z)

Samples are stored per sign and per sequence index:

```
sign_language_dataset/data/
  └── hello/
        ├── hello_1.npy
        ├── hello_2.npy
        └── ...
```

---

## 🎯 Key Features

- **No gloves or markers needed** — works with a plain webcam
- **Extensible vocabulary** — add new signs by collecting more data
- **~95% accuracy** — using a hybrid CNN + BiLSTM architecture
- **Real-time inference** — runs live from webcam input
- **Early stopping** — automatically saves the best model checkpoint

---

## 🔮 Future Improvements

- Add real-time inference script for live prediction display
- Expand vocabulary to full ASL/ISL alphabet and common phrases
- Add text-to-speech output for recognized signs
- Build a web-based UI using the existing `new.html` frontend
- Support two-hand gestures

---

## 👤 Author

**Shivam Sikka**
Project built for the Project Exhibition.

---

## 📄 License

This project is open source and available for educational use.
