import torch
import os
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset

# Constants
MODEL_PATH = 'valence_arousal_model.pt'
BERT_MODEL = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class ValenceArousalModel(nn.Module):
    def __init__(self):
        super(ValenceArousalModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.regression = nn.Linear(768, 2)  # Assuming the last layer outputs two values: Valence and Arousal

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regression(pooled_output)

# Function to load the model
def load_model(model_path):
    model = ValenceArousalModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Function to predict Valence and Arousal from a text file
def predict_from_file(model, file_path):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            inputs = tokenizer(line.strip(), max_length=512, padding='max_length', truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids'].to(DEVICE)
            attention_mask = inputs['attention_mask'].to(DEVICE)
            with torch.no_grad():
                prediction = model(input_ids, attention_mask)
            prediction = (prediction - 1) / 4
            results.append(prediction.cpu().numpy())
            print(f"text: {line}, valence and arousal: {prediction}")
    return results

# Main execution logic
if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_PATH)

    # File path containing text for prediction
    text_file_path = 'prompts.txt'  # Ensure this path is correctly set to where your input text file is located.

    # Predict Valence and Arousal
    predictions = predict_from_file(model, text_file_path)

    # Print predictions
    print("Predictions (Valence, Arousal):")
    for pred in predictions:
        print(pred)
