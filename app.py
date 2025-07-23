import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

print("--- RUNNING LATEST GCP CODE v1 ---")

# --- Basic Flask App Setup ---
app = Flask(__name__)
CORS(app) # Allow all origins by default, suitable for a public API

# --- GLOBAL CONFIGURATION ---
# Securely get the API key from an environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=api_key)

embedding_model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-flash-latest')
html_folder_path = '.' # Look for HTML files in the root directory
collection_name = "healthcare_ai_docs"

# --- IN-MEMORY DATABASE SETUP for Google Cloud Run ---
# We use the standard client because Cloud Run's filesystem is not persistent
client = chromadb.Client()
collection = None

# --- Function to load, chunk, and create the database ---
def initialize_database():
    """
    Reads HTML files, chunks the text, generates embeddings,
    and creates an in-memory ChromaDB collection.
    """
    global collection
    print("Initializing new in-memory database...")

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

    if not all_text_chunks:
        print("CRITICAL: No text chunks found. Cannot initialize database.")
        return

    print("Generating embeddings with Gemini AI...")
    response = genai.embed_content(model=embedding_model, content=all_text_chunks, task_type="retrieval_document")
    embeddings = response['embedding']

    # Create the collection in memory
    collection = client.create_collection(collection_name)
    collection.add(
        ids=[f"doc_{i}" for i in range(len(all_text_chunks))],
        embeddings=embeddings,
        documents=all_text_chunks
    )
    print(f"✅ In-memory database initialized successfully with {collection.count()} documents!")

# --- Initialize the database on startup ---
initialize_database()

# --- Main Chatbot Logic Function ---
def get_chatbot_response(user_query):
    """
    Takes a user query, generates an embedding, queries the database,
    and returns a response from the Gemini chat model.
    """
    global collection
    if not collection:
        return "Sorry, the chatbot database is not available at the moment."

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
    """Handles incoming chat messages."""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    bot_response = get_chatbot_response(user_message)
    return jsonify({"response": bot_response})

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """A simple endpoint to check if the server is running."""
    return jsonify({"status": "ok", "database_initialized": collection is not None})

# --- Run the App (for local testing, not used by Gunicorn) ---
if __name__ == '__main__':
    # This part is for local development. Gunicorn runs the app directly in production.
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
