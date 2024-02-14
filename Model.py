#Used in Tensorflow Model
import numpy as np
import tensorflow as tf
import random

#Usde to for Contextualisation and Other NLP Tasks.
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#Other
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

print("Processing the Intents.....")
with open('intents.json') as json_data:
    intents = json.load(json_data)

print("Processing the Intents.....")
with open('intents.json') as json_data:
    intents = json.load(json_data)

#Text Normalization
import re

def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

#Handling Typos and Spelling Variations
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spelling(text):
    corrected_text = []
    for word in text.split():
        corrected_text.append(spell.correction(word))
    return ' '.join(corrected_text)

words = []
classes = []
documents = []
responses = []
ignore_words = ['?']
print("Looping through the Intents to Convert them to words, classes, documents and ignore_words.......")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    for response in intent['responses']:
      if intent['responses'] not in responses:
            responses.append(intent['responses'])

print("Stemming, Lowering and Removing Duplicates.......")
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print("Creating the Data for our Model.....")
training = []
output = []
print("Creating an List (Empty) for Output.....")
output_empty = [0] * len(classes)

print("Creating Traning Set, Bag of Words for our Model....")
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

print("Shuffling Randomly and Converting into Numpy Array for Faster Processing......")
random.shuffle(training)
training = np.array(training)

print("Creating Train and Test Lists.....")
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Building Neural Network for Out Chatbot to be Contextual....")
print("Resetting graph data....")
tf.compat.v1.reset_default_graph()

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Shuffling Randomly and Converting into Numpy Array for Faster Processing......")
random.shuffle(training)
training = np.array(training)

print("Creating Train and Test Lists.....")
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Building Neural Network for Out Chatbot to be Contextual....")
print("Resetting graph data....")
tf.compat.v1.reset_default_graph()


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Setup TensorBoard callback
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tflearn_logs', histogram_freq=1)

# Print model summary
model.summary()

# Start training
#print("Training....")

# Number of epochs
epochs = 100  # You can adjust this based on how well your model is learning

# Batch size
batch_size = 5  # Adjust based on the size of your dataset and computational resources

# Training the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)

print("Saving the Model.......")
model.save('model.tflearn')

print("Pickle is also Saved..........")
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

print("Saving the Model.......")
model.save('model.tflearn')

print("Pickle is also Saved..........")
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

print("Loading Pickle.....")
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


with open('intents.json') as json_data:
    intents = json.load(json_data)

print("Loading the Model......")
# Load the saved model
loaded_model = tf.keras.models.load_model('model.tflearn')

def clean_up_sentence(sentence):
    # It Tokenize or Break it into the constituents parts of Sentense.
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming means to find the root of the word.
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Return the Array of Bag of Words: True or False and 0 or 1 for each word of bag that exists in the Sentence
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

ERROR_THRESHOLD = 0.25
print("ERROR_THRESHOLD = 0.25")

def classify(sentence):
    # Prediction or To Get the Posibility or Probability from the Model
    results = model.predict([bow(sentence, words)])[0]
    # Exclude those results which are Below Threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # Sorting is Done because heigher Confidence Answer comes first.
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1])) #Tuppl -> Intent and Probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # That Means if Classification is Done then Find the Matching Tag.
    if results:
        # Long Loop to get the Result.
        while results:
            for i in intents['intents']:
                # Tag Finding
                if i['tag'] == results[0][0]:
                    # Random Response from High Order Probabilities
                    return print(random.choice(i['responses']))

            results.pop(0)

while True:
    input_data = input("You- ")

    # Ensure that the input_data is not empty
    if input_data.strip() != "":
        if input_data.lower() == "exit":
            print("Bot: Goodbye!")
            break  # Exit the loop

        # Preprocess the input_data to match the expected input shape
        input_bag = bow(input_data, words)

        # Call the model to predict the intent
        results = model.predict(np.array([input_bag]))[0]

        # Filter intents based on the ERROR_THRESHOLD
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]

        # Sort intents by confidence score
        results.sort(key=lambda x: x[1], reverse=True)

        # Get the most confident intent and find a matching response
        if results:
            # Extract the top intent's index
            top_intent_index = results[0][0]
            top_intent = classes[top_intent_index]

            # Find the intent in the intents.json data
            response_found = False
            for intent in intents['intents']:
                if intent['tag'] == top_intent:
                    # Select a random response from the intent's responses
                    response = random.choice(intent['responses'])
                    print("Bot:", response)
                    response_found = True
                    break

            if not response_found:
                print("Bot: I'm not sure how to respond to that.")
        else:
            print("Bot: I'm sorry, I don't understand.")