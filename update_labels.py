import os
import re
import pandas as pd
import argparse

def identify_word_and_speaker(filepath):
    """
    Identify the base word and speaker from the audio filepath.
    
    Args:
        filepath: Full path to the audio file
    
    Returns:
        tuple: (identified_word, speaker)
    """
    base_words = ["adi", "alayon", "buwas", "gabi", "gab-i", "hain", "kaon", 
                 "marasa", "maupay", "ngaran", "tagpira", "aga"]
    
    # Extract filename and folder name
    filename = os.path.basename(filepath)
    folder_name = os.path.basename(os.path.dirname(filepath))
    
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # First try to identify speaker from folder name
    speaker_names = ["che", "mig", "gab", "josh", "nicole", "sam"]
    speaker = None
    
    # Check if folder name directly indicates the speaker
    for speaker_name in speaker_names:
        if speaker_name.lower() in folder_name.lower():
            speaker = speaker_name
            break
    
    # If speaker not identified from folder, fall back to filename
    if not speaker:
        if "che" in name:
            speaker = "che"
        elif "mig" in name:
            speaker = "mig"
        elif "gab" in name and "gabi" not in name and "gab-i" not in name:
            speaker = "gab"
        elif "josh" in name:
            speaker = "josh"
        elif "nicole" in name:
            speaker = "nicole"
        elif "sam" in name:
            speaker = "sam"
        else:
            speaker = "default"
    
    # Identify base word
    identified_word = None
    for word in base_words:
        # Check if filename starts with the base word
        if name.startswith(word) or name.lower().startswith(word):
            identified_word = word
            break
    
    # Handle special case for gab-i vs gabi
    if identified_word == "gabi" and "gab-i" in name:
        identified_word = "gab-i"
    
    return identified_word, speaker

def update_labels_csv(audio_dir, csv_path='labels.csv', output_path=None):
    """
    Update the labels.csv file with word and speaker information.
    
    Args:
        audio_dir: Directory containing audio files
        csv_path: Path to the existing labels.csv
        output_path: Path to save the updated CSV (if None, overwrites the original)
    """
    if output_path is None:
        output_path = csv_path
    
    # Read existing CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing CSV with {len(df)} entries")
    except FileNotFoundError:
        df = pd.DataFrame(columns=['filename', 'text'])
        print("Creating new CSV file")
    
    # Walk through audio_dir and all subdirectories to get all audio files
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create new dataframe with updated information
    updated_data = []
    for filepath in audio_files:
        word, speaker = identify_word_and_speaker(filepath)
        if word:
            # Use only the filename (not full path) in the CSV
            filename = os.path.basename(filepath)
            updated_data.append({
                'filename': filename,
                'text': word,
                'speaker': speaker
            })
        else:
            print(f"Could not identify word for {filepath}, skipping")
    
    updated_df = pd.DataFrame(updated_data)
    
    # Merge with existing data or use the new data
    if len(df) > 0:
        # Create a combined DataFrame while avoiding duplicates
        combined_df = pd.concat([df, updated_df]).drop_duplicates(subset=['filename'], keep='last')
        
        # Make sure speaker column exists in the old data
        if 'speaker' not in df.columns:
            # For rows from the original DF without speaker info, update them
            for idx, row in df.iterrows():
                # We don't have folder info for existing entries, so use filename only
                if row['filename'] in combined_df['filename'].values:
                    word, speaker = identify_word_and_speaker(row['filename'])
                    if word:
                        combined_df.loc[combined_df['filename'] == row['filename'], 'speaker'] = speaker
    else:
        combined_df = updated_df
    
    # Merge "gab-i" entries into "gabi"
    combined_df.loc[combined_df['text'] == 'gab-i', 'text'] = 'gabi'
    print("Merged 'gab-i' entries into 'gabi'")
    
    # Save updated CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Updated CSV saved to {output_path} with {len(combined_df)} entries")
    
    # Print summary
    print("\nSpeaker distribution:")
    print(combined_df['speaker'].value_counts())
    
    print("\nWord distribution:")
    print(combined_df['text'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update labels.csv with word and speaker information')
    parser.add_argument('--audio_dir', type=str, default='audio', help='Directory containing audio files')
    parser.add_argument('--csv_path', type=str, default='labels.csv', help='Path to existing labels.csv')
    parser.add_argument('--output', type=str, default=None, help='Path to save updated CSV (default: overwrite original)')
    
    args = parser.parse_args()
    update_labels_csv(args.audio_dir, args.csv_path, args.output)