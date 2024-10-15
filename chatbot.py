import random
import json
import torch
from model import NeuralNet
from nltk_script import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE , weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name ="Zeinab"


def get_response(msg):
    # Tokenize and prepare the input
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)  # Ensure it is on the correct device

    # Get the model's prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate the softmax probabilities
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()]

    # Optional: Check the confidence level
    if confidence < 0.75:  # Adjust the threshold as necessary
        return "I'm not sure about that. Can you please clarify?"

    # Find the corresponding response
    for intent in intents["intents"]:
        if tag == intent["tag"]:  # Fixing the condition here
            return random.choice(intent['responses'])
    
    return "I do not understand...."
