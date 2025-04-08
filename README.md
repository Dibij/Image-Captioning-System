
# 🖼️ Image Captioning System with Attention Mechanism

This project generates natural language captions for images using a deep learning model with an attention mechanism. It combines computer vision (InceptionV3 for feature extraction) with NLP (LSTM with attention) to describe what's happening in an image.

---

## 📁 Dataset

- **Source:** MS COCO Dataset (`captions_train2017.json`)
- **Images Used:** First 1000 training images for demonstration purposes.
- **Captions:** Preprocessed using NLTK (tokenization, punctuation removal, lowercasing).

---

## 🧠 Model Overview

- **Feature Extractor:** InceptionV3 (pretrained on ImageNet)
- **Caption Decoder:** LSTM with Bahdanau Attention
- **Special Tokens:** `<start>`, `<end>`, `<unk>`
- **Tokenization:** Keras Tokenizer
- **Max Sequence Length:** Automatically calculated from data

---

## 🏗️ Architecture

1. **Image Input:** Extracted features from InceptionV3.
2. **Caption Input:** Tokenized and embedded sequences.
3. **Attention Layer:** Bahdanau attention applied on image features.
4. **LSTM Decoder:** Predicts the next word in the sequence.
5. **Output:** Word probabilities over vocabulary.

---

## 🏋️‍♂️ Training

- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam (LR = 0.0001)
- **Epochs:** 10
- **Batch Size:** 32
- **Train/Val Split:** 80/20

---

## 📦 Output

- **Model Saved As:** `attention_image_captioning_model.h5`
- **Prediction Function:** Can caption any new image using the trained model.

---

## ▶️ Sample Prediction

```python
Generated caption: a group of people standing around a kitchen
```

(From image: `000000000009.jpg`)

---

## 🔧 Setup Instructions

```bash
pip install tensorflow pillow scikit-learn nltk
python -m nltk.downloader punkt
```

Ensure your COCO dataset is properly placed under:

```
/content/coco_dataset/
├── annotations/
│   └── captions_train2017.json
└── images/
    └── train2017/
```

---

## 📌 Notes

- ⚠️ Only a small subset of COCO is used for demo/training.
- Model structure can be improved further with beam search or Transformer-based encoders.
