import os
# Fix for OpenMP conflict (must be before torch import)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from train import FusionEmotionModel, MultimodalDataset

def test():
    print("Testing Multimodal Fusion Model...")
    # Loading the master dataset
    csv_path = '../../tess_multimodal_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    test_loader = DataLoader(MultimodalDataset(val_df), batch_size=16, shuffle=False)
    
    model = FusionEmotionModel()
    
    model_path = 'fusion_model.pth'
    if not os.path.exists(model_path):
         print(f"Error: Model not found at {model_path}. Please run train.py first.")
         return

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    
    print("Testing in progress...")
    with torch.no_grad():
        for speech, ids, mask, labels in test_loader:
            outputs = model(speech, ids, mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print("-" * 30)
    print(f'Final Multimodal Fusion Accuracy: {accuracy:.2f}%')
    print("-" * 30)

if __name__ == "__main__":
    test()