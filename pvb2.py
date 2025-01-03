import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus = []
responses = []
for intent in intents:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern.lower())
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        corpus.append(" ".join(filtered_tokens))
        responses.append(intent['responses'])

# Vectorizer and model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.arange(len(corpus))

model = MultinomialNB()
model.fit(X, y)

# Chatbot response function
def chatbot_response(user_input):
    user_tokens = word_tokenize(user_input.lower())
    filtered_user_tokens = [lemmatizer.lemmatize(word) for word in user_tokens if word.isalnum() and word not in stop_words]
    user_vector = vectorizer.transform([" ".join(filtered_user_tokens)])
    prediction = model.predict(user_vector)
    return np.random.choice(responses[prediction[0]])

# Streamlit sidebar for menu
menu = ["Home", "Conversation History", "Intents Used", "About"]
selection = st.sidebar.selectbox("Menu", menu)

# Title
st.title("Hospital Chatbot")
st.write("Ask me something!")

# Initialize conversation history in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Define function to display conversation history
def show_conversation_history():
    st.header("Conversation History")
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

# Define function to show intents used
def show_intents_used():
    st.header("Intents Used")
    for intent in intents:
        st.write(f"**{intent['tag']}**")
        st.write(f"Patterns: {', '.join(intent['patterns'])}")
        st.write(f"Responses: {', '.join(intent['responses'])}")
        st.write("---")

# Define function to show about section
def show_about():
    st.header("About")
    st.write("""
    This is a hospital chatbot powered by Natural Language Processing (NLP). It uses machine learning models to respond to user queries based on predefined intents and patterns.
    The chatbot can answer questions related to hospital services, appointments, and more.
    """)

# Show the selected menu content
if selection == "Home":
    st.write("### Chat with me!")
    # User input logic
    user_input = st.text_input("You:", "", key="input_box", placeholder="Type your message here...")

    if user_input:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "text": user_input})

        # Generate chatbot response
        response = chatbot_response(user_input)
        st.session_state.conversation.append({"role": "bot", "text": response})

    # Display conversation messages
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

elif selection == "Conversation History":
    show_conversation_history()
elif selection == "Intents Used":
    show_intents_used()
elif selection == "About":
    show_about()
