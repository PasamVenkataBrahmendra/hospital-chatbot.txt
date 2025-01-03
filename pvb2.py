import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json

# Load the intents from a JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Prepare the corpus and responses from intents
corpus = []
responses = []
for intent in intents:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern.lower())
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        corpus.append(" ".join(filtered_tokens))
        responses.append(intent['responses'])

# Vectorizer and classifier setup
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.arange(len(corpus))

model = MultinomialNB()
model.fit(X, y)

# Function to get chatbot response
def chatbot_response(user_input):
    user_tokens = word_tokenize(user_input.lower())
    filtered_user_tokens = [lemmatizer.lemmatize(word) for word in user_tokens if word.isalnum() and word not in stop_words]
    user_vector = vectorizer.transform([" ".join(filtered_user_tokens)])
    prediction = model.predict(user_vector)
    return np.random.choice(responses[prediction[0]])

# Sidebar with menu options
st.sidebar.title("Menu")
menu_option = st.sidebar.radio("Select an option", ("Conversation History", "Intents Used", "About"))

# Display menu content
if menu_option == "Conversation History":
    st.sidebar.write("Conversation history will be displayed here.")
elif menu_option == "Intents Used":
    st.sidebar.write("List of intents used in the chatbot.")
elif menu_option == "About":
    st.sidebar.write("This is a Hospital Chatbot built using NLP techniques and Streamlit.")

# Main content area
st.title("Hospital Chatbot")
st.write("Ask me something!")

# Initialize conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input at the bottom of the page
user_input = st.text_input("You:", "", key="input_box")

if user_input:
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "text": user_input})
    
    # Generate chatbot response
    response = chatbot_response(user_input)
    st.session_state.conversation.append({"role": "bot", "text": response})

# Display conversation with chat bubbles
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.markdown(f"""
        <div style='background-color: #e8f5e9; border-radius: 10px; padding: 10px; margin: 10px 0; width: fit-content; max-width: 80%; text-align: left;'>
            <b>You:</b> {message['text']}
        </div>
        """, unsafe_allow_html=True)
    elif message["role"] == "bot":
        st.markdown(f"""
        <div style='background-color: #e3f2fd; border-radius: 10px; padding: 10px; margin: 10px 0; width: fit-content; max-width: 80%; text-align: left; margin-left: auto;'>
            <b>Chatbot:</b> {message['text']}
        </div>
        """, unsafe_allow_html=True)

# Always scroll to the bottom of the conversation
st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)
