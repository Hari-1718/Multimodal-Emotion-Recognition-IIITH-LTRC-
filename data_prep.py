import os
import pandas as pd

# I'm using the dataset folder inside my project directory
DATASET_PATH = 'dataset'

data = []

print("Starting data extraction...")

# The TESS dataset has separate folders for each emotion/actor.
# I need to loop through them to build a single CSV.
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    
    # making sure it's actually a folder before entering
    if os.path.isdir(folder_path):
        # The folder names are like 'OAF_Fear', so I split by '_' and take the last part
        emotion_label = folder.split('_')[-1].lower()
        
        # Now going through every audio file in this emotion folder
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                full_path = os.path.abspath(os.path.join(folder_path, file))
                
                # The filename format is usually OAF_word_emotion.wav
                # I need the middle part to get the text.
                parts = file.split('_')
                if len(parts) >= 2:
                    word = parts[1]
                    # As per the docs, they say "Say the word [word]"
                    transcript = f"Say the word {word}"
                    
                    data.append({
                        'file_path': full_path,
                        'text': transcript,
                        'label': emotion_label
                    })

# Save everything to a CSV so I don't have to parse folders every time
df = pd.DataFrame(data)
df.to_csv('tess_multimodal_data.csv', index=False)
print(f"Done! Saved {len(df)} samples to tess_multimodal_data.csv.")