import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# 1. Configuration
DATA_PATH = '../../tess_multimodal_data.csv'
# Hyperparameters - tuned these a bit
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20  # 20 seems enough for convergence on this dataset

# 2. Robust Data Loader
class SpeechDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        # Handling the label issues (some folders had typos like 'surprised')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
        self.label_map = {emotion: i for i, emotion in enumerate(self.emotions)}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['file_path']
        label_str = self.df.iloc[idx]['label'].lower()
        
        # Clean label variations
        if 'surpris' in label_str:
            label_str = 'pleasant_surprise'
        
        label = self.label_map.get(label_str, 4) # Default to neutral if error
        
        # Feature Extraction: Using MFCCs as they are standard for speech emotion
        # I chose 40 MFCCs to capture enough detail
        audio, sr = librosa.load(path, duration=3.0, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Time steps need to be fixed for the LSTM
        # Padding with zeros if shorter, trimming if longer
        if mfcc.shape[1] < 100:
            mfcc = np.pad(mfcc, ((0,0), (0, 100 - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :100]
            
        return torch.tensor(mfcc, dtype=torch.float32).T, torch.tensor(label, dtype=torch.long)

# 3. Model Architecture
class SpeechEmotionModel(nn.Module):
    def __init__(self):
        super(SpeechEmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, 7) 

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.classifier(hn[-1])
        return out

# 4. Training Function
def train():
    print("Checking for CSV file...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Could not find CSV at {os.path.abspath(DATA_PATH)}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples from CSV.")
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = DataLoader(SpeechDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    
    model = SpeechEmotionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Speech Model Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for mfccs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), 'speech_model.pth')
    print("SUCCESS: Model saved as speech_model.pth")

if __name__ == "__main__":
    train()