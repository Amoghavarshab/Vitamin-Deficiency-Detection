import nltk
import json
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('intents1.json', 'r') as file:
    intents = json.load(file)

# Load the trained model
model = load_model('chatbot_model.h5')

# Preprocessing function for the intents data
def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', ',', '.']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add to documents
            documents.append((word_list, intent['tag']))
            # Add to classes if not already there
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and remove duplicates
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
    words = sorted(set(words))
    classes = sorted(set(classes))

    return words, classes, documents

# Get preprocessed data
words, classes, documents = preprocess_data(intents)

# Function to clean up the sentence for prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words for the input sentence
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Predict class of the sentence
def predict_class(sentence, model):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Function to predict the intent and return a response
def predict_and_respond(message):
    # Predict the class of the message
    ints = predict_class(message, model)
    
    if ints:
        # Get the response based on the predicted intent
        res = get_response(ints, intents)
        return res
    else:
        return "Sorry, I didn't understand that."

# Example usage
if __name__ == "__main__":
    test_input = input("enter the symtoms")
    response = predict_and_respond(test_input)
    #print(f"Input: {test_input}")
    print(f"Response: {response}")
