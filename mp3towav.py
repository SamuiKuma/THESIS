from pydub import AudioSegment
import os

def m4a_to_wav(m4a_file_path, wav_file_path):
    # Load the M4A file
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    
    # Export as WAV file
    audio.export(wav_file_path, format="wav")

def batch_convert_m4a_to_wav(m4a_directory,wav_directory):
    for filename in os.listdir(m4a_directory):
        if filename.endswith(".m4a"):
            m4a_file_path = os.path.join(m4a_directory, filename)
            wav_file_path = os.path.join(wav_directory, os.path.splitext(filename)[0] + ".wav")
            m4a_to_wav(m4a_file_path, wav_file_path)
            print(f"Converted {m4a_file_path} to {wav_file_path}")

# Example usage
m4a_directory = r"D:\lstm\audio_conversions\m4a"  # Replace with your directory containing M4A files
wav_directory= r"D:\lstm\audio_conversions\wav"  # Replace with your directory to save WAV files
batch_convert_m4a_to_wav(m4a_directory,wav_directory)