import os
import time
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Classification Inference API", version="1.0.0")

# Global variables for model and word index
model = None
word_index = None
vocab_size = 10000
max_len = 200

# Input schema
class PredictRequest(BaseModel):
    texts: List[str]

@app.on_event("startup")
async def startup_event():
    global model, word_index
    logger.info("Initializing ML Server...")

    # Load Model
    model_path = 'models/text_classification_model_latest.keras'
    fallback_path = 'text_classification_model.keras'
    
    try:
        if os.path.exists(model_path):
            logger.info(f"Loading versioned model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
        elif os.path.exists(fallback_path):
            logger.info(f"Loading fallback model from {fallback_path}...")
            model = tf.keras.models.load_model(fallback_path)
        else:
            raise FileNotFoundError("No trained model found. Please run train.py first.")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    # Load Word Index
    logger.info("Loading IMDB word index map...")
    raw_word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in raw_word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    logger.info("Vocabulary loaded.")

def encode_texts(texts: List[str]):
    tokens_list = []
    for text in texts:
        words = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = []
        for word in words:
            idx = word_index.get(word, 2)
            if idx >= vocab_size:
                idx = 2
            tokens.append(idx)
        tokens_list.append(tokens)
    return pad_sequences(tokens_list, maxlen=max_len, padding='post', truncating='post')

@app.middleware("http")
async def log_requests_and_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000
    
    # Log latency
    logger.info(f"Path: {request.url.path} - Method: {request.method} - Latency: {process_time_ms:.2f} ms")
    
    response.headers["X-Process-Time"] = str(process_time_ms)
    return response

@app.post("/predict")
async def predict_sentiment(request: PredictRequest):
    if not model:
        return {"error": "Model not loaded properly."}
    
    # Log the incoming prediction stats batch size
    batch_size = len(request.texts)
    logger.info(f"Received prediction request for {batch_size} sample(s).")
    
    # Process inputs
    encoded_inputs = encode_texts(request.texts)
    
    # Make predictions
    predictions = model.predict(encoded_inputs, verbose=0)
    
    results = []
    for i in range(batch_size):
        pred_value = float(predictions[i][0])
        sentiment = "Positive" if pred_value > 0.5 else "Negative"
        confidence = pred_value if sentiment == "Positive" else (1.0 - pred_value)
        
        # Log prediction stat internally
        text_preview = request.texts[i][:50] + "..." if len(request.texts[i]) > 50 else request.texts[i]
        logger.info(f"Predicted '{sentiment}' (Confidence: {confidence:.4f}) for input: '{text_preview}'")
        
        results.append({
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_probability": pred_value
        })
        
    return {"results": results}
