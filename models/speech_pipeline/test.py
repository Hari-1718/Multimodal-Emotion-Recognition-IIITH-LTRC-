import os
# Fix for OpenMP conflict (must be before torch import)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# Importing from my train script to keep things consistent
from train import SpeechEmotionModel, SpeechDataset

def test():
    print("Loading test data...")
    # I'm using the master CSV file
    csv_path = '../../tess_multimodal_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    # Use the same split logic as training
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    test_loader = DataLoader(SpeechDataset(val_df), batch_size=32, shuffle=False)
    
    model = SpeechEmotionModel()
    
    model_path = 'speech_model.pth'
    if not os.path.exists(model_path):
         print(f"Error: Model not found at {model_path}. Please run train.py first.")
         return
         
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    
    print("Testing in progress...")
    with torch.no_grad():
        for mfccs, labels in test_loader:
            outputs = model(mfccs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print("-" * 30)
    print(f'Final Speech Model Accuracy: {accuracy:.2f}%')
    print("-" * 30)

if __name__ == "__main__":
    test()