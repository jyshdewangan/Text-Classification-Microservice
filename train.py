import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, precision_recall_fscore_support

def main():
    print("Loading data...")
    # Setting vocabulary size and max sequence length
    vocab_size = 10000
    max_len = 200
    batch_size = 64
    epochs = 10

    # Load data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    # Pad sequences for uniform length
    print("Padding sequences...")
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

    # Optimize training and inference using tf.data
    # Applying batching and prefetching for performance optimization
    print("Creating tf.data pipelines...")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build the Neural Network Model
    # Applying Embedding and Sequence modeling (Bidirectional LSTM)
    print("Building model...")
    model = Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=128),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Model tuning with Callbacks: Early Stopping and Learning Rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-5)

    print("Training model...")
    # Train the model
    # To keep it simple and fast, we'll run just a few epochs, but it's configured for real training
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[early_stopping, reduce_lr]
    )

    print("Evaluating model...")
    # Get predictions
    y_pred_probs = model.predict(test_dataset)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Evaluate model using precision, recall, and F1-score metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    
    # Model versioning: Save the model with a simple version tag (timestamp)
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_model_path = f'models/text_classification_model_v1_{timestamp}.keras'
    model.save(versioned_model_path)
    print(f"Versioned model saved to: {versioned_model_path}")
    
    # Save a latest reference for the FastAPI endpoint
    latest_model_path = 'models/text_classification_model_latest.keras'
    model.save(latest_model_path)
    print(f"Latest model reference updated at: {latest_model_path}")

if __name__ == "__main__":
    main()
