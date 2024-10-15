import json
from nltk_script import tokenize, stem, bag_of_words  # Import the functions from nltkscript.py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

# Load the intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each intent
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    # Loop through each pattern in the current intent
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Ignore certain characters
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)  # Corrected this line to use the tags list
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Create a custom dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    # Get item by index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Return the total number of samples
    def __len__(self):
        return self.n_samples

# Set batch size
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

# Initialize the dataset and DataLoader
dataset = ChatDataset()

# Use num_workers=0 to avoid multiprocessing issues on Windows
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)  # Convert labels to LongTensor

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss={loss.item():.4f}')
    
print(f'Final loss, Loss = {loss.item():.4f}')



data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words"   : all_words,
    "tags"        : tags

}


FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
