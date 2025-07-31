import streamlit as st
import os
import nltk
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# 👉 Try local nltk_data path first
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)
else:
    nltk.download("punkt")
    nltk.download("stopwords")

# ✅ Load sentence transformer model only once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Preprocessing ---
def preprocess(text):
    stop_words = set(stopwords.words("english"))
    sentences = sent_tokenize(text)
    cleaned = []

    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w not in stop_words and w not in string.punctuation]
        cleaned.append(" ".join(words))

    return sentences, cleaned

# --- Matching Logic ---
def get_most_relevant_sentence(user_input, cleaned_sentences, original_sentences, embeddings):
    query_embedding = embedder.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_idx = scores.argmax().item()

    if scores[best_idx] < 0.4:
        return "🤔 I'm not sure how to answer that right now."

    return original_sentences[best_idx]

# --- Load Knowledge Base ---
faq_path = os.path.join(os.path.dirname(__file__), "..", "data", "customer_faq.txt")
with open(faq_path, "r", encoding="utf-8") as f:
    text = f.read()

original_sentences, cleaned_sentences = preprocess(text)
embeddings = embedder.encode(cleaned_sentences, convert_to_tensor=True)

# --- Streamlit UI ---
st.title("🤖 Fresh Support Chatbot")
st.write("Ask me anything about your order, refund, shipping, or returns.")

user_input = st.text_input("You:")

if user_input:
    response = get_most_relevant_sentence(user_input, cleaned_sentences, original_sentences, embeddings)
    st.success(f"💬 {response}")
