import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Constants
DATA_FILE = 'emobank.csv'
BERT_MODEL = 'bert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MODEL_PATH = 'valence_arousal_model.pt'

# Data Loading
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Dataset Class
class EmoBankDataset(Dataset):
    def __init__(self, texts, valences, arousals, tokenizer, max_len=512):
        self.texts = texts
        self.valences = valences
        self.arousals = arousals
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        valence = self.valences[idx]
        arousal = self.arousals[idx]
        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'valence': torch.tensor(valence, dtype=torch.float),
            'arousal': torch.tensor(arousal, dtype=torch.float)
        }

# Model Definition
class ValenceArousalModel(nn.Module):
    def __init__(self):
        super(ValenceArousalModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.regression = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regression(pooled_output)

# Training Function with per-iteration loss tracking
def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    iteration_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = torch.stack((batch['valence'], batch['arousal']), dim=1).to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            iteration_losses.append(loss.item())

        val_loss, _ = evaluate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Last Batch Loss: {iteration_losses[-1]:.4f}, Val Loss: {val_loss:.4f}')

    plot_losses(iteration_losses, 'Iteration', 'Loss', 'Training Loss per Iteration')

# Modified Plotting Function
def plot_losses(losses, xlabel, ylabel, title):
    plt.plot(losses, label='Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig("loss_per_iteration.png")

# Evaluation Function
def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = torch.stack((batch['valence'], batch['arousal']), dim=1).to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader), mean_squared_error(targets.cpu(), outputs.cpu())

def predict_vad_from_file(model, file_path, tokenizer, device):
    model.eval()
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    predictions = []
    for line in lines:
        inputs = tokenizer(line.strip(), max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        predictions.append(outputs.cpu().numpy())
        print(f"text: {line}, valence and arousal: {predictions}")
    
    return predictions

# Main
if __name__ == "__main__":
    data = load_data(DATA_FILE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    text_file_path = "prompts.txt"

    # Splitting the data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Resetting indices
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Creating datasets
    # 这里感觉
    train_dataset = EmoBankDataset(train_data['text'], train_data['V'], train_data['A'], tokenizer)
    val_dataset = EmoBankDataset(val_data['text'], val_data['V'], val_data['A'], tokenizer)
    test_dataset = EmoBankDataset(test_data['text'], test_data['V'], test_data['A'], tokenizer)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValenceArousalModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train(model, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS)
    torch.save(model.state_dict(), 'valence_arousal_model.pt')

    vad_predictions = predict_vad_from_file(model, text_file_path, tokenizer, device)
    # print("VAD Predictions:", vad_predictions)
