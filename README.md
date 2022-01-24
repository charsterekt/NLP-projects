# NLP Research Projects

This repository contains a few projects I undertook to gain a better understanding of PyTorch and NLP techniques. PyTorch has long been on my list of frameworks to learn because it is far easier to use than TensorFlow due to its Pythonic structure and it has caught up in terms of supporting libraries and frameworks and models. Here are the projects I have currently worked on: <br>

## Intent Based Chatbot

A simple intent based chatbot that gets its training data from a JSON file of intents and is trained on a simple three layer feed forward PyTorch NN. The bot isn't very smart, but the beauty of intent classifier bots is that they only grow with the amount of data provided to them. If I had chosen to use a premade intent dataset I might have achieved a better result, but as it is, the bot serves as proof of concept. I have also demonstrated how external libraries or programs/scripts of any sort can be tied to the intent classifier, thus allowing us to do pretty much anything as long as we can associate it with an intent.

## Sentiment Analysis using BERT

Bi-directional Encoder Representations of Transformers, BERT is all the rage in the NLP world ever since Google released the paper on it. And thanks to amazing open source libraries like HuggingFace Transformers, anyone can leverage the power of BERT for their tasks. In this project, I have used an app review dataset from the Google Play Store in tandem with BERT from HuggingFace to classify positive, neutral, and negative intents. The model performs reasonably well given the limited data. I also used matplotlib and seaborn to visualize the data at various stages. It served as a great learning experience.
