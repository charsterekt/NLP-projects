import random 
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from modules import jokes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('DumbBot/intents.json', 'r') as f:
    intents = json.load(f)

FILE = "Dumbass/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Evaluation mode

# Data Structure for responses

responses = {
    "greeting": [
        "Hey :-)",
        "Hello there!",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?",
        "At your service!"
    ],
    "goodbye": [
        "See you later!",
        "Have a nice day",
        "Bye! Come back again soon."
    ],
    "thanks": [
        "Happy to help!", 
        "Any time!", 
        "My pleasure"
    ],
    "items": [
        "We sell coffee and tea", 
        "We have coffee and tea"
    ],
    "payments": [
        "We accept VISA, Mastercard and Paypal",
        "We accept most major credit cards, and Paypal"
    ],
    "delivery": [
        "Delivery takes 2-4 days",
        "Shipping takes 2-4 days"
    ],
    "funny": jokes.get_joke()
}

# Chat feature
bot_name = "DumbBot"
print("Let's chat! Type 'quit' to exit")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)

    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Adding softmax probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"Category: {tag}, Confidence: {prob.item() * 100}")

    if prob.item() > 0.50:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(responses[tag])}")

    else:
        print(f"{bot_name}: Sorry, i'm dumb")