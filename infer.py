import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def test_model():
    model_path = 'text_classification_model.keras'
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first by running 'python train.py'")
        return
    
    print("Loading IMDB word index...")
    word_index = imdb.get_word_index()
    
    # Adjust word indices according to IMDB convention
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  
    word_index["<UNUSED>"] = 3

    vocab_size = 10000
    max_len = 200

    def encode_text(text):
        words = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = []
        for word in words:
            idx = word_index.get(word, 2)
            # Cap indices at vocab_size
            if idx >= vocab_size:
                idx = 2
            tokens.append(idx)
        return pad_sequences([tokens], maxlen=max_len, padding='post', truncating='post')

    sample_texts = [
        "This movie was absolutely wonderful, I loved every minute. A true masterpiece of cinema.", 
        "Terrible acting and the plot made no sense. Complete waste of time, do not watch.",
        "It was an okay film, not great but not terrible either. The acting was acceptable.",
        "I was on the edge of my seat the entire time. A true masterpiece.",
        "Such a boring experience. I fell asleep halfway through.",
        "The visual effects were stunning, but the storyline lacked depth.",
        "A brilliant performance by the lead actor, totally deserves an Oscar.",
        "One of the worst movies I have ever seen. Save your money.",
        "It started off slow but the climax was incredible! Highly recommended.",
        "Utter garbage. I regret spending time on this."
    ]

    print("\n--- Predictions ---")
    for text in sample_texts:
        encoded = encode_text(text)
        prediction = model.predict(encoded, verbose=0)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment} (Confidence Score: {prediction:.4f})\n")

if __name__ == "__main__":
    test_model()
