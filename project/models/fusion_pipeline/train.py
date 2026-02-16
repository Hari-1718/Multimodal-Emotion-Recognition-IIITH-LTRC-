import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# 1. Configuration
DATA_PATH = '../../tess_multimodal_data.csv'
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5

# 2. Multimodal Dataset (Speech + Text)
class MultimodalDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
        self.label_map = {emotion: i for i, emotion in enumerate(self.emotions)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- Speech Preprocessing ---
        audio_path = self.df.iloc[idx]['file_path']
        audio, sr = librosa.load(audio_path, duration=3.0, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 100:
            mfcc = np.pad(mfcc, ((0,0), (0, 100 - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :100]
        
        # --- Text Preprocessing ---
        text = self.df.iloc[idx]['text']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
        
        # --- Label Handling ---
        label_str = self.df.iloc[idx]['label'].lower()
        if 'surpris' in label_str: label_str = 'pleasant_surprise'
        label = self.label_map.get(label_str, 4)

        return (torch.tensor(mfcc, dtype=torch.float32).T, 
                encoding['input_ids'].flatten(), 
                encoding['attention_mask'].flatten(), 
                torch.tensor(label))

# 3. Fusion Architecture
class FusionEmotionModel(nn.Module):
    def __init__(self):
        super(FusionEmotionModel, self).__init__()
        # 1. Speech Branch (LSTM)
        self.speech_lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2, batch_first=True)
        
        # 2. Text Branch (BERT)
        self.text_bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 3. Fusion Layer
        # Concatenating the outputs: 128 from Speech + 768 from Text
        self.classifier = nn.Sequential(
            nn.Linear(128 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 7) # Output 7 emotions
        )

    def forward(self, speech_x, text_ids, text_mask):
        # Pass speech through LSTM
        _, (hn, _) = self.speech_lstm(speech_x)
        speech_feat = hn[-1] # Take the last hidden state
        
        # Pass text through BERT
        text_outputs = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        text_feat = text_outputs.pooler_output
        
        # Late Fusion: Combining features
        fused = torch.cat((speech_feat, text_feat), dim=1)
        return self.classifier(fused)

# 4. Training
def train():
    df = pd.read_csv(DATA_PATH)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    loader = DataLoader(MultimodalDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    
    model = FusionEmotionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting Multimodal Fusion Training...")
    for epoch in range(EPOCHS):
        model.train()
        for speech, ids, mask, labels in loader:
            optimizer.zero_grad()
            outputs = model(speech, ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), 'fusion_model.pth')
    print("SUCCESS: Multimodal fusion model saved.")

if __name__ == "__main__":
    train()