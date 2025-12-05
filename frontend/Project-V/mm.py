import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import json
import random
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
with open('intents1.json', 'r') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing
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

words, classes, documents = preprocess_data(intents)

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

# Function to load patterns from 'patterns.txt'
def load_patterns_from_file():
    with open('patterns.txt', 'r') as file:
        patterns = file.read().split(', ')
    return patterns

patterns_from_file = load_patterns_from_file()

# Chatbot response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

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

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Chat
print("Chatbot is ready! Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break

    # Check if the input is in patterns.txt
    if message.lower() not in [pattern.lower() for pattern in patterns_from_file]:
        print("Chatbot:Could not identify deficiency. Please consult a nearby doctor.")
        continue
    
    ints = predict_class(message, model)

    if ints:
        res = get_response(ints, intents)
        print(f"Chatbot: {res}")
        
    else:
        print("Chatbot: Data can't be generated.")
