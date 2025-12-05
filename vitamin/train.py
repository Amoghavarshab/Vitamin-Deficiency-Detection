# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import numpy as np
# import json
# import random
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer
# import pickle

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load model and data
# model = load_model('chatbot_model.h5')
# intents = json.loads(open('intents1.json', encoding='utf-8').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

# # Function to generate summary
# def generate_summary(text):
#     stopWords = set(stopwords.words("english"))
#     words = word_tokenize(text.lower())
    
#     # Creating a frequency table
#     freqTable = {}
#     for word in words:
#         if word not in stopWords and word.isalpha():
#             freqTable[word] = freqTable.get(word, 0) + 1

#     sentences = sent_tokenize(text)
#     sentenceValue = {}

#     for sentence in sentences:
#         sentenceWords = word_tokenize(sentence.lower())
#         sentenceValue[sentence] = sum(freqTable.get(word, 0) for word in sentenceWords if word in freqTable)

#     if not sentenceValue:
#         return "Summary could not be generated."

#     # Calculating the average value of a sentence
#     average = sum(sentenceValue.values()) / len(sentenceValue)
    
#     # Storing sentences into our summary
#     summary = ' '.join(sentence for sentence in sentences if sentenceValue.get(sentence, 0) > 1.0 * average)

#     return summary if summary.strip() else "Summary could not be generated."

# # Clean up the sentence
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# # Convert sentence to bag of words
# def bow(sentence, words):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence, model):
#     p = bow(sentence, words)
#     res = model.predict(np.array([p]))[0]
    
#     ERROR_THRESHOLD = 0.50  # Set a threshold for acceptable predictions
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
#     results.sort(key=lambda x: x[1], reverse=True)
    
#     if not results:
#         return [{"intent": "data_not_found", "probability": "0"}]
    
#     return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
#     return return_list

# def get_response(ints, intents_json):
#     if not ints or ints[0].get('intent') == 'data_not_found':
#         return "Data can't be generated for the provided input."
    
#     tag = ints[0]['intent']
#     list_of_intents = intents_json.get('intents', [])
    
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
    
#     return "Dataset not trained."

# # Generate chatbot response
# def chatbot_response(msg):
#     ints = predict_class(msg, model)
#     res = get_response(ints, intents)
#     return res

# # Main loop for console input
# while True:
#     text = input("Enter symptoms: ").strip()
#     if text.lower() == 'exit':
#         break

#     summary = generate_summary(text)
#     response = chatbot_response(text)

# ##    print("\nSummary:")
# ##    print(summary)
#     print("\nChatbot Response:")
#     print(response)
#     print("\n" + "="*50 + "\n")


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import json
import random
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents1.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to generate summary
def generate_summary(text):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    
    # Creating a frequency table
    freqTable = {}
    for word in words:
        if word not in stopWords and word.isalpha():
            freqTable[word] = freqTable.get(word, 0) + 1

    sentences = sent_tokenize(text)
    sentenceValue = {}

    for sentence in sentences:
        sentenceWords = word_tokenize(sentence.lower())
        sentenceValue[sentence] = sum(freqTable.get(word, 0) for word in sentenceWords if word in freqTable)

    if not sentenceValue:
        return "Summary could not be generated."

    # Calculating the average value of a sentence
    average = sum(sentenceValue.values()) / len(sentenceValue)
    
    # Storing sentences into our summary
    summary = ' '.join(sentence for sentence in sentences if sentenceValue.get(sentence, 0) > 1.0 * average)

    return summary if summary.strip() else "Summary could not be generated."

# Clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    
    ERROR_THRESHOLD = 0.50  # Set a threshold for acceptable predictions
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    if not results:
        return [{"intent": "data_not_found", "probability": "0"}]
    
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    if not ints or ints[0].get('intent') == 'data_not_found':
        return "Could not identify deficiency Please Consult Near by Doctor"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json.get('intents', [])
    
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    
    return "Dataset not trained."

# Generate chatbot response
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res
