import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# Config
DATA_PATH = '../../tess_multimodal_data.csv'
BATCH_SIZE = 32
EPOCHS = 5 # BERT learns fast, so 5 epochs is plenty
LEARNING_RATE = 2e-5 # Standard LR for BERT fine-tuning

# 2. Text Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.texts = texts
        self.labels = labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
        self.label_map = {emotion: i for i, emotion in enumerate(self.emotions)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_str = self.labels[idx].lower()
        if 'surpris' in label_str: label_str = 'pleasant_surprise'
        
        label = self.label_map.get(label_str, 4)
        
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
        return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), torch.tensor(label)

# 3. Contextual Modelling Architecture (BERT)
class TextEmotionModel(nn.Module):
    def __init__(self):
        super(TextEmotionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 7)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # We use the pooler_output (CLS token) for classification
        return self.classifier(outputs.pooler_output)

# 4. Training
def train():
    df = pd.read_csv(DATA_PATH)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    
    dataset = TextDataset(train_df['text'].values, train_df['label'].values)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = TextEmotionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting Text Model Training...")
    model.train()
    for epoch in range(EPOCHS):
        for ids, mask, labels in loader:
            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), 'text_model.pth')
    print("SUCCESS: Text model saved.")

if __name__ == "__main__":
    train()