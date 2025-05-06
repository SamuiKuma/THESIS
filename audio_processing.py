# audio_processing.py
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import random
from scipy.signal import resample

def extract_features(file_path, max_pad_len=100, audio=None, sr=None):
    """
    Extract audio features (MFCCs, spectral features) from an audio file or pre-loaded audio.
    Handles short files and ensures consistent feature dimensions.
    """
    try:
        # Load audio file if not provided
        if audio is None:
            audio, sr = librosa.load(file_path, sr=None)
        
        # Skip files that are too short or empty
        if len(audio) < sr * 0.1:  # Skip if less than 0.1 seconds
            print(f"Warning: {file_path} is too short ({len(audio)/sr:.3f}s). Skipping...")
            return None
            
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Extract zero crossing rate safely
        zero_crossing = librosa.feature.zero_crossing_rate(audio)
        
        # Safe standardization with fallback for zero variance
        try:
            if np.std(zero_crossing) > 0:
                zero_crossing = (zero_crossing - np.mean(zero_crossing)) / np.std(zero_crossing)
            else:
                zero_crossing = zero_crossing - np.mean(zero_crossing)  # Just center if std is zero
        except:
            # If standardization fails, use the raw values
            pass
            
        # Ensure both features have the same time dimension for concatenation
        if mfccs.shape[1] != zero_crossing.shape[1]:
            # Resize to match the shorter one
            min_len = min(mfccs.shape[1], zero_crossing.shape[1])
            mfccs = mfccs[:, :min_len]
            zero_crossing = zero_crossing[:, :min_len]
        
        # Stack features
        features = np.vstack((mfccs, zero_crossing))
        
        # Transpose to get time as the first dimension
        features = features.T
        
        # Pad or truncate to ensure consistent length
        if features.shape[0] < max_pad_len:
            # Pad with zeros if too short
            pad_width = ((0, max_pad_len - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width, mode='constant')
        else:
            # Truncate if too long
            features = features[:max_pad_len, :]
            
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def augment_audio(file_path):
    """Generate augmented versions of an audio file's features with advanced techniques."""
    try:
        # Load original audio
        audio, sr = librosa.load(file_path, sr=None)
        
        # Skip if file is too short
        if len(audio) < sr * 0.1:
            return []
            
        augmented_features = []
            
        # 1. Time stretching with more variety
        stretch_rates = [0.85, 0.9, 1.1, 1.15]
        for rate in stretch_rates:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            features = extract_features(None, audio=stretched, sr=sr)
            if features is not None:
                augmented_features.append(features)
                
        # 2. Pitch shifting with more variety
        semitones = [-3, -2, 2, 3]
        for n_steps in semitones:
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            features = extract_features(None, audio=shifted, sr=sr)
            if features is not None:
                augmented_features.append(features)
        
        # 3. Add noise with different intensities
        noise_factors = [0.003, 0.005, 0.01]
        for noise_factor in noise_factors:
            noisy_audio = audio + noise_factor * np.random.normal(0, 1, len(audio))
            features = extract_features(None, audio=noisy_audio, sr=sr)
            if features is not None:
                augmented_features.append(features)
                
        # 4. Time masking (silence portions of the audio)
        mask_width = int(len(audio) * 0.1)  # Mask 10% of the audio
        mask_start = np.random.randint(0, len(audio) - mask_width)
        masked_audio = np.copy(audio)
        masked_audio[mask_start:mask_start+mask_width] = 0
        features = extract_features(None, audio=masked_audio, sr=sr)
        if features is not None:
            augmented_features.append(features)
        
        return augmented_features
        
    except Exception as e:
        print(f"Error augmenting {file_path}: {str(e)}")
        return []

def is_valid_audio(file_path):
    try:
        audio_segment = AudioSegment.from_file(file_path)
        duration_ms = len(audio_segment)
        volume_db = audio_segment.dBFS
        sample_rate = audio_segment.frame_rate
        if duration_ms < 200 or duration_ms > 5000:
            return False
        if volume_db < -45:
            return False
        if sample_rate != 16000:
            return False
        return True
    except Exception as e:
        print(f"Error validating audio: {e}")
        return False

def enhance_audio_for_waray(audio_segment):
    try:
        audio = audio_segment.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = audio.high_pass_filter(80)
        audio = audio.low_pass_filter(7500)
        audio = audio.normalize()
        audio = audio + 10
        return audio
    except Exception as e:
        print(f"Error in audio enhancement: {e}")
        return audio_segment
