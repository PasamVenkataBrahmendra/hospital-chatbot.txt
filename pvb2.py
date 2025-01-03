import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import json

with open('intents1.json', 'r') as file:
    intents = json.load(file)['intents']

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

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

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.arange(len(corpus))

model = MultinomialNB()
model.fit(X, y)

def chatbot_response(user_input):
    user_tokens = word_tokenize(user_input.lower())
    filtered_user_tokens = [lemmatizer.lemmatize(word) for word in user_tokens if word.isalnum() and word not in stop_words]
    user_vector = vectorizer.transform([" ".join(filtered_user_tokens)])
    prediction = model.predict(user_vector)
    return np.random.choice(responses[prediction[0]])

st.title("Hospital Chatbot")
st.write("Ask me something!")

# Create two columns: one for chat and another for sidebar content
col1, col2 = st.columns([3, 1])

# Chat history on the left side
with col1:
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Display conversation history
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.markdown(f"""
            <div style='background-color: #e8f5e9; border-radius: 10px; padding: 10px; margin: 10px 0; width: fit-content; max-width: 80%; text-align: left;'>{message['text']}</div>
            """, unsafe_allow_html=True)
        elif message["role"] == "bot":
            st.markdown(f"""
            <div style='background-color: #e3f2fd; border-radius: 10px; padding: 10px; margin: 10px 0; width: fit-content; max-width: 80%; text-align: left; margin-left: auto;'>{message['text']}</div>
            """, unsafe_allow_html=True)

    # Add user input at the bottom
    user_input = st.text_input("You:", "", key="input_box")

    if user_input:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "text": user_input})

        # Generate chatbot response
        response = chatbot_response(user_input)
        st.session_state.conversation.append({"role": "bot", "text": response})

# Sidebar on the right for additional sections (conversation history, intents used, about)
with col2:
    st.subheader("Conversation History")
    for message in st.session_state.conversation:
        st.markdown(f"- {message['role'].capitalize()}: {message['text']}")

    st.subheader("Intents Used")
    used_intents = [intent['tag'] for intent in intents]
    st.markdown(f"- {', '.join(used_intents)}")

    st.subheader("About")
    st.write("""
        This is a Hospital Chatbot powered by machine learning.
        You can ask general medical questions and the bot will provide responses based on predefined intents.
        Created using Streamlit and NLP technologies.
    """)
