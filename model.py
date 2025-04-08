import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Dense, Embedding, LSTM, Input, Concatenate, Lambda, Reshape
from tensorflow.keras.models import Model
import nltk
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# ======================
# 1. LOAD AND PREPARE DATA
# ======================
with open('/content/coco_dataset/annotations/annotations/captions_train2017.json', 'r') as f:
    captions_data = json.load(f)

# ======================
# 2. FEATURE EXTRACTION
# ======================
print("Loading InceptionV3 model...")
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((299, 299))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = inception_model.predict(img, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

print("Extracting image features...")
features_dict = {}
for img_info in captions_data['images'][:1000]:  # Using first 1000 for demo
    image_path = os.path.join('/content/coco_dataset/images/train2017', img_info['file_name'])
    if os.path.exists(image_path):
        features = extract_features(image_path)
        if features is not None:
            features_dict[img_info['id']] = features

# ======================
# 3. CAPTION PREPROCESSING
# ======================
def preprocess_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    tokens = word_tokenize(caption)
    return ' '.join(tokens)

print("Preprocessing captions...")
all_captions = []
image_ids = []
for ann in captions_data['annotations']:
    if ann['image_id'] in features_dict:
        preprocessed_caption = preprocess_caption(ann['caption'])
        all_captions.append(preprocessed_caption)
        image_ids.append(ann['image_id'])



# ======================
# 4. TOKENIZATION
# ======================
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

# Add special tokens
tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")



# ======================
# 5. SEQUENCE CREATION
# ======================
max_length = max(len(seq) for seq in tokenizer.texts_to_sequences(all_captions)) 
print(f"Maximum sequence length: {max_length}")

def encode_caption(caption):
    caption = '<start> ' + caption + ' <end>'
    seq = tokenizer.texts_to_sequences([caption])[0]
    return seq

sequences = [encode_caption(cap) for cap in all_captions]
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')



# ======================
# 6. DATA PREPARATION
# ======================
image_features = []
captions = []
for img_id, seq in zip(image_ids, padded_sequences):
    if img_id in features_dict:
        image_features.append(features_dict[img_id])
        captions.append(seq)

image_features = np.array(image_features)
captions = np.array(captions)



# ======================
# 7. TRAIN-VAL SPLIT
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    image_features, captions, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")



# ======================
# 8. ATTENTION MODEL
# ======================
class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_attention_model(vocab_size, max_length, embedding_dim=256, units=256):
    # Image encoder
    image_input = Input(shape=(2048,))
    image_dense = Dense(units, activation='relu')(image_input)
    image_features = Reshape((1, units))(image_dense)
    
    # Caption decoder
    caption_input = Input(shape=(max_length,))
    embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    
    # Initialize LSTM and attention
    lstm = LSTM(units, return_sequences=True, return_state=True)
    attention = BahdanauAttention(units)
    
    # Initial states
    hidden = Dense(units)(image_dense)
    cell = Dense(units)(image_dense)
    
    outputs = []
    
    # Process each time step
    for t in range(max_length):
        context_vector, _ = attention(image_features, hidden)
        lstm_input = Concatenate()([embedding[:, t, :], context_vector])
        lstm_input = Reshape((1, embedding_dim + units))(lstm_input)
        
        output, hidden, cell = lstm(lstm_input, initial_state=[hidden, cell])
        output = Dense(vocab_size, activation='softmax')(output[:, -1, :])
        outputs.append(output)
    
    outputs = Lambda(lambda x: tf.stack(x, axis=1))(outputs) 
    return Model(inputs=[image_input, caption_input], outputs=outputs)

# Build and compile the model
model = build_attention_model(vocab_size, max_length - 1)  
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Verify shapes before training
print("\nShape verification:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train input shape: {y_train[:, :-1].shape}")
print(f"y_train target shape: {y_train[:, 1:].shape}")



# ======================
# 9. MODEL TRAINING
# ======================
history = model.fit(
    x=[X_train, y_train[:, :-1]],  # Input: Image features, captions without last word
    y=y_train[:, 1:],              # Target: Captions without first word
    validation_data=([X_val, y_val[:, :-1]], y_val[:, 1:]),
    epochs=10,
    batch_size=32,
    verbose=1
)



# ======================
# 10. SAVE MODEL
# ======================
model.save('attention_image_captioning_model.h5')
print("Model saved with attention mechanism!")



# ======================
# 11. PREDICTION FUNCTION
# ======================
def predict_caption(image_path, model, tokenizer, max_length=None):
    # Extract features
    features = extract_features(image_path)
    if features is None:
        return "Could not process image"
    
    features = np.expand_dims(features, axis=0)
    in_text = '<start>'
    
    # Get the model's expected sequence length from its input shape
    model_sequence_length = model.input_shape[1][1]
    
    for _ in range(model_sequence_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad to the model's expected length
        sequence = pad_sequences([sequence], maxlen=model_sequence_length, padding='post')
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat[0, -1, :])
        word = tokenizer.index_word.get(yhat, '')
        
        if word == '<end>':
            break
            
        in_text += ' ' + word
    
    return in_text.replace('<start>', '').strip()

# Test prediction
test_image = "/content/coco_dataset/images/train2017/000000000009.jpg"
print("\nGenerated caption:", predict_caption(test_image, model, tokenizer))


