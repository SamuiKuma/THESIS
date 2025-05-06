# train_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import random
from tqdm import tqdm
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_generator(audio_files, labels, batch_size=32):
    """
    Generator to yield batches of audio features and labels.
    """
    while True:
        indices = np.random.permutation(len(audio_files))
        for start_idx in range(0, len(audio_files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_x = []
            batch_y = []
            for idx in batch_indices:
                features = extract_features(audio_files[idx])
                if features is not None:
                    batch_x.append(features)
                    batch_y.append(labels[idx])
            yield np.array(batch_x), np.array(batch_y)

def load_dataset(csv_path, audio_dir, augment_count=2):
    """Load and preprocess the dataset with VM source tracking."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Original dataset size: {len(df)}")
        
        # Log VM distribution
        if 'speaker' in df.columns:
            logger.info("\nSpeaker distribution:")
            logger.info(df['speaker'].value_counts())
        
        # Target Waray words - expanded to include all specified words
        target_words = [
            "adi", "aga", "alayon", "buwas", "gabi", "gab-i", 
            "hain", "kaon", "marasa", "maupay", "ngaran", "tagpira"
        ]
        
        df = df[df['text'].isin(target_words)]
        logger.info(f"Dataset size after filtering for target words: {len(df)}")
        
        # Load audio features and labels
        X = []
        y = []
        
        from audio_processing import extract_features, augment_audio
        
        # Create a mapping of all audio files in audio_dir and its subdirectories
        audio_files_map = {}
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    # Store both the full path and just the filename
                    audio_files_map[file] = os.path.join(root, file)
        
        logger.info(f"Found {len(audio_files_map)} audio files in directory structure")
        
        # Track successfully loaded files
        loaded_files = 0
        missing_files = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            filename = row['filename']
            
            # Try to find the file in our audio_files_map
            if filename in audio_files_map:
                audio_file = audio_files_map[filename]
            else:
                # Search through all subdirectories
                found = False
                for root, _, files in os.walk(audio_dir):
                    potential_path = os.path.join(root, filename)
                    if os.path.exists(potential_path):
                        audio_file = potential_path
                        found = True
                        break
                
                if not found:
                    # Still not found, use the default path
                    audio_file = os.path.join(audio_dir, filename)
                
            if not os.path.exists(audio_file):
                logger.warning(f"Audio file not found: {filename}")
                missing_files += 1
                continue
                
            features = extract_features(audio_file)
            if features is not None:
                X.append(features)
                y.append(row['text'])
                loaded_files += 1
                
                # Data augmentation based on speaker
                if augment_count > 0:
                    augmented_features = augment_audio(audio_file)
                    for aug_feature in augmented_features:
                        if aug_feature is not None:
                            X.append(aug_feature)
                            y.append(row['text'])
        
        logger.info(f"Successfully loaded {loaded_files} files, {missing_files} files not found")
        
        # Convert to arrays
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Final dataset size: {len(X)}")
        logger.info(f"Feature shape: {X.shape}")
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Save encoder classes
        np.save('encoder_classes.npy', encoder.classes_)
        
        return X, y_encoded, encoder.classes_
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def augment_training_data(X, y, augment_factor=2):
    """Augment training data with time stretching, pitch shifting, and noise."""
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        # Add original
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Time stretching
        stretch_rates = [0.9, 1.1]
        for rate in stretch_rates[:augment_factor]:
            stretched = librosa.effects.time_stretch(X[i], rate=rate)
            if len(stretched) >= X.shape[1]:
                stretched = stretched[:X.shape[1]]
            else:
                stretched = np.pad(stretched, ((0, X.shape[1] - len(stretched)), (0, 0)))
            X_aug.append(stretched)
            y_aug.append(y[i])
        
        # Add noise
        if augment_factor > 1:
            noise = np.random.normal(0, 0.005, X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

def build_model(input_shape, num_classes):
    """Build an improved LSTM model with attention for speech recognition."""
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # LSTM layers with batch normalization
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second LSTM layer
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Reshape((-1,))(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention_expanded = tf.keras.layers.RepeatVector(128)(attention)
    attention_expanded = tf.keras.layers.Permute((2, 1))(attention_expanded)
    
    # Apply attention
    context = tf.keras.layers.Multiply()([x, attention_expanded])
    
    # Final LSTM layer
    x = tf.keras.layers.LSTM(64)(context)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Save attention weights for later analysis
    model.attention_weights = attention
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_waray_model(csv_path='labels.csv', audio_dir='audio', epochs=50, batch_size=32):
    """Train the Waray speech recognition model."""
    try:
        # Load dataset
        X, y, classes = load_dataset(csv_path, audio_dir, augment_count=3)
        logger.info(f"Classes: {classes}")
        
        # Augment training data
        X, y = augment_training_data(X, y, augment_factor=2)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        num_classes = len(classes)
        model = build_model(input_shape, num_classes)
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save model
        model.save('waray_speech_model.keras')
        logger.info("Model saved successfully")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting Waray speech recognition model training")
    # Update the audio_dir parameter to point to the correct location
    train_waray_model(audio_dir=r'D:\lstm_prof\audio_wav')
    logger.info("Training completed")
