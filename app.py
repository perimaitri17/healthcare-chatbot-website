import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Basic Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- GLOBAL CONFIGURATION ---
genai.configure(api_key="AIzaSyBDBs9WhF46leK7tDfZA_UqznXjr-c0_Kg")
embedding_model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-pro-latest')
html_folder_path = 'html_files'

# --- PERSISTENT DATABASE SETUP ---
# Use PersistentClient to save the database to disk on Render
# Render provides a persistent disk at '/var/data'
db_path = "/var/data/chroma"
client = chromadb.PersistentClient(path=db_path)
collection = None  # We will initialize this below

# --- Function to load, chunk, and create the database ---
def initialize_database():
    global collection
    print("Initializing new database...")

    # 1. Load and Chunk Data from HTML files
    all_text_chunks = []
    print(f"Reading HTML files from: {html_folder_path}")
    for filename in os.listdir(html_folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(html_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                text = soup.get_text(separator='\n', strip=True)
                chunks = [chunk for chunk in text.split('\n') if len(chunk.strip()) > 10]
                all_text_chunks.extend(chunks)
    print(f"Found {len(all_text_chunks)} total text chunks.")

    # 2. Generate Embeddings
    print("Generating embeddings with Gemini AI...")
    response = genai.embed_content(model=embedding_model, content=all_text_chunks, task_type="retrieval_document")
    embeddings = response['embedding']

    # 3. Create and populate the collection
    collection = client.create_collection("healthcare_ai_docs")
    collection.add(
        ids=[f"doc_{i}" for i in range(len(all_text_chunks))],
        embeddings=embeddings,
        documents=all_text_chunks
    )
    print("✅ New database initialized and stored successfully!")

# --- Check for Database on Startup ---
try:
    # Try to get the existing collection
    collection = client.get_collection("healthcare_ai_docs")
    print("✅ Connected to existing database.")
except ValueError:
    # If it doesn't exist, the above line will throw a ValueError
    print("Database not found. Starting a new one...")
    initialize_database()


# --- Main Chatbot Logic Function ---
def get_chatbot_response(user_query):
    query_embedding = genai.embed_content(model=embedding_model, content=user_query, task_type="retrieval_query")['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_context = "\n\n".join(results['documents'][0])
    prompt = f"""
    You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
    If the context doesn't contain the answer, say "I do not have enough information to answer that."

    CONTEXT:
    {retrieved_context}

    QUESTION:
    {user_query}
    """
    final_answer = chat_model.generate_content(prompt)
    return final_answer.text


# --- API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    bot_response = get_chatbot_response(user_message)
    return jsonify({"response": bot_response})


# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
