import os
import re
import shutil

def rename_files(directory):
    """
    Rename files from "Grass Parkway number" to "ngaran_che (number)" format.
    
    Args:
        directory: Path to the directory containing the files
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    # Get list of files in directory
    files = os.listdir(directory)
    print(f"Found {len(files)} files in directory")
    
    # Print first few files to see the actual naming pattern
    if files:
        print("Sample filenames:")
        for filename in files[:5]:
            print(f"  - {filename}")
    
    # Updated pattern to match "Grass Parkway number" without parentheses
    pattern = re.compile(r'grass\s+parkway\s+(\d+)', re.IGNORECASE)
    
    # Keep track of renamed files
    renamed_count = 0
    
    # Process each file
    for filename in files:
        # Check if file matches the pattern
        match = pattern.search(filename)
        if match:
            # Extract the number
            number = match.group(1)
            
            # Create new filename - note we're adding parentheses in the new name
            new_filename = f"maupay_che ({number}){os.path.splitext(filename)[1]}"
            
            # Full paths for old and new filenames
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
        # For debugging - print files that don't match
        else:
            print(f"No match for: {filename}")
    
    print(f"\nRenamed {renamed_count} files.")

if __name__ == "__main__":
    # Directory containing the files
    directory = r"D:\lstm_prof\audio_wav\vms\maupay\che"
    rename_files(directory)