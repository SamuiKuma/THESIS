# This file was renamed from speech_recognition.py to speech_recognition_utils.py to avoid import conflicts.
import os
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from pydub import AudioSegment
from g2p_en import G2p
import logging
from audio_processing import extract_features, enhance_audio_for_waray, is_valid_audio

# Add these imports at the top of your file
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import librosa
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self, model_path=None, encoder_path=None, confidence_threshold=0.2):
        """Initialize speech processor with model loading and configuration."""
        self.model = None
        
        # Set default paths, but prioritize passed-in paths
        if model_path is None:
            # Try multiple possible locations
            possible_model_paths = [
                "waray_speech_model.keras",                  # Current directory
                "d:/lstm_prof/waray_speech_model.keras",     # Original path
                os.path.join(os.path.dirname(__file__), "waray_speech_model.keras")  # Same folder as this script
            ]
            # Use the first one that exists
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                model_path = "waray_speech_model.keras"  # Default if none found
        
        # Same approach for encoder path
        if encoder_path is None:
            possible_encoder_paths = [
                "encoder_classes.npy",                 # Current directory
                os.path.join(os.path.dirname(__file__), "encoder_classes.npy"),  # Same folder as this script
                "d:/lstm_prof/encoder_classes.npy"     # Try absolute path
            ]
            for path in possible_encoder_paths:
                if os.path.exists(path):
                    encoder_path = path
                    break
            else:
                encoder_path = "encoder_classes.npy"  # Default if none found
        
        try:
            # Try to load the model
            logger.info(f"Attempting to load model from {model_path}")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                print(f"ERROR: Model file not found at {model_path}")
                print(f"Current working directory: {os.getcwd()}")
                print("Please train the model first using train_model.py or specify the correct path.")
                # We'll continue initialization but with limited functionality
                self.model = None
            else:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded model from {model_path}")
            
            # Try to load encoder classes, with fallback to default list
            try:
                logger.info(f"Attempting to load encoder classes from {encoder_path}")
                if not os.path.exists(encoder_path):
                    logger.warning(f"Encoder classes file not found at {encoder_path}")
                    print(f"WARNING: Encoder file not found at {encoder_path}")
                    print(f"Current working directory: {os.getcwd()}")
                    raise FileNotFoundError(f"Encoder classes file not found: {encoder_path}")
                
                self.encoder_classes = np.load(encoder_path)
                print(f"Successfully loaded encoder classes from {encoder_path}")
            except FileNotFoundError:
                logger.warning(f"Encoder classes file not found: {encoder_path}. Using default Waray word list.")
                # Default list of Waray words as fallback
                self.encoder_classes = np.array([
                    "adi", "aga", "alayon", "buwas", "gabi", "gab-i", 
                    "hain", "kaon", "marasa", "maupay", "ngaran", "tagpira"
                ])
                
            self.confidence_threshold = confidence_threshold
            self.g2p = G2p()
            
            if self.model is not None:
                logger.info("Successfully initialized SpeechProcessor")
            else:
                logger.warning("SpeechProcessor initialized with limited functionality (no model loaded)")
                
        except Exception as e:
            logger.error(f"Error initializing SpeechProcessor: {str(e)}")
            raise

        # Initialize Waray phoneme mappings - expanded for all 10 words
        self.waray_phonemes = {
            'ngaran': ['ŋ', 'a', 'ɾ', 'a', 'n'],
            'gab-i': ['g', 'a', 'b', '-', 'i'],
            'maupay': ['m', 'a', 'u', 'p', 'a', 'y'],
            'aga': ['a', 'g', 'a'],
            'adi': ['a', 'd', 'i'],
            'alayon': ['a', 'l', 'a', 'y', 'o', 'n'],
            'buwas': ['b', 'u', 'w', 'a', 's'],
            'gabi': ['g', 'a', 'b', 'i'],
            'hain': ['h', 'a', 'i', 'n'],
            'kaon': ['k', 'a', 'o', 'n'],
            'marasa': ['m', 'a', 'ɾ', 'a', 's', 'a'],
            'tagpira': ['t', 'a', 'g', 'p', 'i', 'ɾ', 'a']
        }

    def predict_speech(self, audio_file, expected_word=None):
        """Predict speech with enhanced error handling and detailed feedback."""
        try:
            if not is_valid_audio(audio_file):
                logger.warning(f"Invalid audio file: {audio_file}")
                return None, None, None

            # Extract features from the audio file
            features = extract_features(audio_file)
            if features is None:
                return None, None, None

            # Make prediction
            features = np.expand_dims(features, axis=0)
            prediction = self.model.predict(features, verbose=0)
            
            confidence = np.max(prediction)
            if confidence < self.confidence_threshold:
                logger.warning(f"Low confidence prediction: {confidence:.2f}")
                return None, None, None

            # Get predicted word
            predicted_index = np.argmax(prediction)
            predicted_word = self.encoder_classes[predicted_index]

            # Calculate phoneme-level confidence if expected word matches
            phoneme_confidence = None
            if expected_word and predicted_word.lower() == expected_word.lower():
                phoneme_confidence = self._analyze_phoneme_confidence(features, predicted_word)
                
                # Generate specific feedback for problematic phonemes
                if phoneme_confidence:
                    weak_phonemes = []
                    for phoneme, score in phoneme_confidence.items():
                        if score < 0.7:  # Threshold for acceptable pronunciation
                            weak_phonemes.append((phoneme, score))
                    
                    if weak_phonemes:
                        logger.info(f"Pronunciation issues detected in phonemes: {[p[0] for p in weak_phonemes]}")
                        
                        # Map phonemes to syllables for more intuitive feedback
                        if expected_word.lower() in self.waray_phonemes:
                            syllables = self._map_phonemes_to_syllables(expected_word.lower())
                            problem_syllables = self._identify_problem_syllables(weak_phonemes, syllables)
                            
                            if problem_syllables:
                                syllable_feedback = f"Focus on these syllables: {', '.join(problem_syllables)}"
                                logger.info(syllable_feedback)

            return predicted_word, confidence, phoneme_confidence

        except Exception as e:
            logger.error(f"Error in speech prediction: {str(e)}")
            return None, None, None

    def _map_phonemes_to_syllables(self, word):
        """Map phonemes to syllables for more intuitive feedback."""
        syllable_mapping = {
            'ngaran': ['nga', 'ran'],
            'gab-i': ['gab', 'i'],
            'maupay': ['ma', 'u', 'pay'],
            'aga': ['a', 'ga'],
            'adi': ['a', 'di'],
            'alayon': ['a', 'la', 'yon'],
            'buwas': ['bu', 'was'],
            'gabi': ['ga', 'bi'],
            'hain': ['ha', 'in'],
            'kaon': ['ka', 'on'],
            'marasa': ['ma', 'ra', 'sa'],
            'tagpira': ['tag', 'pi', 'ra']
        }
        
        return syllable_mapping.get(word, [word])  # Return whole word if mapping not found
        
    def _identify_problem_syllables(self, weak_phonemes, syllables):
        """Identify which syllables have pronunciation issues based on phoneme analysis."""
        # Maps phonemes to syllables
        phoneme_to_syllable = {
            'ŋ': 'nga', 'a': ['a', 'ma', 'ga', 'sa', 'ra', 'ha', 'ka', 'tag'],
            'ɾ': ['ran', 'ra'], 'n': ['ran', 'yon', 'in', 'on'],
            'g': ['ga', 'nga'], 'b': ['gab', 'bu'], 'i': ['i', 'di', 'bi', 'pi'],
            'm': ['ma'], 'u': ['u', 'bu'], 'p': ['pay', 'pi'],
            'y': ['pay', 'yon'], 'd': ['di'], 'l': ['la'],
            'o': ['on', 'yon'], 'w': ['was'], 's': ['was', 'sa'],
            'h': ['ha'], 'k': ['ka'], 't': ['tag']
        }
        
        problem_syllables = set()
        for phoneme, score in weak_phonemes:
            if phoneme in phoneme_to_syllable:
                syllable_matches = phoneme_to_syllable[phoneme]
                if isinstance(syllable_matches, list):
                    for s in syllable_matches:
                        if s in syllables:
                            problem_syllables.add(s)
                elif syllable_matches in syllables:
                    problem_syllables.add(syllable_matches)
                    
        return list(problem_syllables)

    def _analyze_phoneme_confidence(self, features, word):
        """Analyze confidence scores for each phoneme using DTW alignment."""
        try:
            if word.lower() in self.waray_phonemes:
                phonemes = self.waray_phonemes[word.lower()]
            else:
                phonemes = self.g2p(word)

            # Extract mel spectrogram features from audio
            mel_features = self._extract_mel_features(features[0])
            
            # Create reference patterns for each phoneme
            phoneme_patterns = self._get_phoneme_reference_patterns(phonemes)
            
            # Use Dynamic Time Warping to align audio with phoneme patterns
            scores = {}
            for i, phoneme in enumerate(phonemes):
                # Calculate DTW between audio features and phoneme pattern
                distance, path = fastdtw(mel_features, phoneme_patterns[phoneme], dist=euclidean)
                
                # Convert distance to confidence score (inverse relationship)
                # Lower distance = higher confidence
                max_distance = 100  # Normalization factor
                confidence = max(0, 1 - (distance / max_distance))
                scores[phoneme] = float(confidence)
            
            # Normalize scores to ensure they're in [0, 1]
            max_score = max(scores.values()) if scores else 1.0
            scores = {k: v/max_score for k, v in scores.items()}

            return scores
        except Exception as e:
            logger.error(f"Error in phoneme analysis: {str(e)}")
            return None

    def _extract_mel_features(self, audio_features):
        """Extract mel spectrogram from audio features for DTW comparison."""
        # Extract the mel portion of the features if it's a combined feature vector
        feature_dim = audio_features.shape[-1]
        mel_dim = min(40, feature_dim)  # Typical mel dimension
        
        # Use the most relevant dimensions for phoneme matching
        return audio_features[:, :mel_dim]

    def _get_phoneme_reference_patterns(self, phonemes):
        """Create or retrieve reference patterns for each phoneme."""
        patterns = {}
        
        # These patterns would ideally be learned from training data
        # For now, we use pre-defined spectral templates for common phonemes
        vowel_shape = np.ones((20, 13)) * 0.8
        consonant_shape = np.ones((20, 13)) * 0.6
        
        # Define basic patterns
        base_patterns = {
            # Vowels with distinct spectral shapes
            'a': vowel_shape * np.array([np.linspace(0.5, 1, 13) for _ in range(20)]),
            'i': vowel_shape * np.array([np.linspace(1, 0.5, 13) for _ in range(20)]),
            'u': vowel_shape * np.array([np.linspace(0.7, 0.9, 13) for _ in range(20)]),
            'e': vowel_shape * np.array([np.linspace(0.8, 0.6, 13) for _ in range(20)]),
            'o': vowel_shape * np.array([np.linspace(0.6, 0.8, 13) for _ in range(20)]),
            
            # Consonants with appropriate characteristics
            'b': consonant_shape * 0.7,
            'd': consonant_shape * 0.75,
            'g': consonant_shape * 0.8,
            'h': consonant_shape * 0.5,
            'k': consonant_shape * 0.9,
            'l': consonant_shape * 0.65,
            'm': consonant_shape * 0.7,
            'n': consonant_shape * 0.7,
            'p': consonant_shape * 0.9,
            'r': consonant_shape * 0.6,
            's': consonant_shape * 0.8,
            't': consonant_shape * 0.85,
            'w': consonant_shape * 0.6,
            'y': consonant_shape * 0.7,
            
            # Special phonemes for Waray
            'ŋ': consonant_shape * 0.8 * np.array([np.linspace(0.7, 0.9, 13) for _ in range(20)]),
            'ɾ': consonant_shape * 0.6 * np.array([np.linspace(0.6, 0.8, 13) for _ in range(20)]),
            '-': np.ones((20, 13)) * 0.3,
        }
        
        # Create patterns for each phoneme in the word
        for phoneme in phonemes:
            if phoneme in base_patterns:
                patterns[phoneme] = base_patterns[phoneme]
            else:
                # Fallback for unknown phonemes
                patterns[phoneme] = np.ones((20, 13)) * 0.5
                
        return patterns

def capture_audio(duration=3, sample_rate=16000):
    """Capture audio from microphone with enhanced preprocessing."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=sample_rate) as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            logger.info("Speak now...")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=duration)
                
                # Save temporary WAV file
                temp_file = "temp_recording.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Enhance audio
                audio_segment = AudioSegment.from_wav(temp_file)
                enhanced_audio = enhance_audio_for_waray(audio_segment)
                enhanced_file = "enhanced_recording.wav"
                enhanced_audio.export(enhanced_file, format="wav")
                
                os.remove(temp_file)
                return enhanced_file
                
            except sr.WaitTimeoutError:
                logger.warning("No speech detected")
                return None
                
    except Exception as e:
        logger.error(f"Error capturing audio: {str(e)}")
        return None

def process_speech(word=None):
    """Main function for speech processing and feedback."""
    try:
        processor = SpeechProcessor()
        
        while True:
            if word:
                print(f"\nPlease pronounce: {word}")
            
            audio_file = capture_audio()
            if not audio_file:
                print("Failed to capture audio. Please try again.")
                continue
                
            try:
                predicted_word, confidence, phoneme_confidence = processor.predict_speech(audio_file, word)
                
                if predicted_word is None:
                    print("Could not recognize speech clearly. Please try again.")
                    continue
                    
                print(f"\nRecognized: {predicted_word}")
                print(f"Confidence: {confidence:.2%}")
                
                if word and phoneme_confidence:
                    print("\nPronunciation Analysis:")
                    for phoneme, score in phoneme_confidence.items():
                        status = "✓" if score > 0.7 else "✗"
                        print(f"{phoneme}: {score:.2%} {status}")
                        
                    avg_confidence = sum(phoneme_confidence.values()) / len(phoneme_confidence)
                    print(f"\nOverall pronunciation accuracy: {avg_confidence:.2%}")
                    
                    if avg_confidence < 0.7:
                        print("\nSuggestions:")
                        # Expanded pronunciation tips for all words
                        word_tips = {
                            'ngaran': [
                                "- Make sure 'ng' is pronounced as one sound",
                                "- Keep the 'r' sound soft"
                            ],
                            'gab-i': [
                                "- Separate 'gab' and 'i' clearly",
                                "- Emphasize the pause between syllables"
                            ],
                            'maupay': [
                                "- Pronounce 'ma-u-pay' as three distinct syllables",
                                "- Make the 'ay' sound clear"
                            ],
                            'aga': [
                                "- Make both 'a' sounds clear",
                                "- Pronounce the 'g' distinctly"
                            ],
                            'adi': [
                                "- Pronounce 'a' clearly and 'di' with a short 'i'",
                                "- Keep the 'd' sound crisp"
                            ],
                            'alayon': [
                                "- Emphasize all three syllables: 'a-la-yon'",
                                "- End with a clear nasal 'n' sound"
                            ],
                            'buwas': [
                                "- Pronounce the 'b' sound at the start clearly",
                                "- Make the 'u-wa' blend smoothly but distinctly"
                            ],
                            'gabi': [
                                "- Emphasize both syllables equally: 'ga-bi'",
                                "- Don't confuse with gab-i; no pause between syllables"
                            ],
                            'hain': [
                                "- Start with a clear 'h' aspiration",
                                "- The 'ain' should sound like 'eye-n' with a nasal ending"
                            ],
                            'kaon': [
                                "- Pronounce the 'k' sharply at the beginning",
                                "- The 'ao' vowel combination should flow smoothly"
                            ],
                            'marasa': [
                                "- Keep the 'r' sound soft, almost like a quick 'd'",
                                "- Emphasize all three syllables: 'ma-ra-sa'"
                            ],
                            'tagpira': [
                                "- Make the 'tag' syllable clear with a distinct 'g'",
                                "- Pronounce 'pi-ra' with equal emphasis on both syllables"
                            ]
                        }
                        
                        for tip in word_tips.get(word.lower(), ["- Try speaking more clearly"]):
                            print(tip)
                
            finally:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            
            if not word:
                break
                
            retry = input("\nTry again? (y/n): ").lower()
            if retry != 'y':
                break
                
    except Exception as e:
        logger.error(f"Error in speech processing: {str(e)}")

if __name__ == "__main__":
    print("Waray Speech Recognition System")
    print("1. Practice Mode (with specific words)")
    print("2. Free Recognition Mode")
    
    choice = input("\nSelect mode (1/2): ")
    
    if choice == "1":
        # Updated word list to include all 10 words
        words = ["adi", "aga", "alayon", "buwas", "gabi", "gab-i", "hain", "kaon", "marasa", "maupay", "ngaran", "tagpira"]
        print("\nAvailable words:", ", ".join(words))
        word = input("Enter word to practice (or press Enter to quit): ").lower()
        if word in words:
            process_speech(word)
        else:
            print("Invalid word selected.")
    else:
        process_speech()