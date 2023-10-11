#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)

# uncomment the following only the first time
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

#Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """if user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

ADDITIONAL_RESPONSES = {
    "how are you": "I'm just a chatbot, but thanks for asking!",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "who created you": "I was created by a team of developers.",
    "bye": "Goodbye! Feel free to come back if you have more questions.",
}

# Update the response function to include additional responses
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = "I am sorry, I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    
    # Check if the user input has an additional response
    if user_response in ADDITIONAL_RESPONSES:
        robo_response = ADDITIONAL_RESPONSES[user_response]
    
    return robo_response

# Chatbot conversation loop
flag = True
print("Julie: My name is Julie. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag:
    user_response = input("You: ")
    user_response = user_response.lower()
    
    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            flag = False
            print("Julie: You're welcome.")
        else:
            if greeting(user_response) is not None:
                print("Julie: " + greeting(user_response))
            else:
                print("Julie: " + response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Julie: Goodbye! Take care.")
