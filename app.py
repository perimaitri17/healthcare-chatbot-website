import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

print("--- GCP Chatbot Service Starting [v2 - Enhanced Prompt] ---")

# --- Basic Flask App Setup ---
app = Flask(__name__)
CORS(app) 

# --- GLOBAL CONFIGURATION ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=api_key)

embedding_model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-flash')
html_folder_path = '.' 
collection_name = "healthcare_ai_docs"

# --- IN-MEMORY DATABASE SETUP ---
client = chromadb.Client()
collection = None

# --- Function to initialize the database ---
def initialize_database():
    global collection
    print("Initializing in-memory vector database...")
    all_text_chunks = []
    for filename in os.listdir(html_folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(html_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                # Add the filename to each chunk for context
                page_content = soup.get_text(separator='\n', strip=True)
                chunks = [f"Context from {filename}:\n{chunk}" for chunk in page_content.split('\n') if len(chunk.strip()) > 20]
                all_text_chunks.extend(chunks)
    print(f"Found {len(all_text_chunks)} total text chunks.")

    if not all_text_chunks:
        print("CRITICAL: No text chunks found.")
        return

    print("Generating embeddings...")
    response = genai.embed_content(model=embedding_model, content=all_text_chunks, task_type="retrieval_document")
    embeddings = response['embedding']
    collection = client.create_collection(collection_name)
    collection.add(
        ids=[f"doc_{i}" for i in range(len(all_text_chunks))],
        embeddings=embeddings,
        documents=all_text_chunks
    )
    print(f"✅ Database initialized with {collection.count()} documents!")

initialize_database()

# --- Main Chatbot Logic Function ---
def get_chatbot_response(user_query):
    global collection
    if not collection:
        return "Sorry, the chatbot database is not initialized."

    # First, check for simple navigation commands
    lower_query = user_query.lower()
    if "go to" in lower_query or "take me to" in lower_query or "navigate to" in lower_query:
        if "safety" in lower_query: return "NAVIGATE_TO_SAFETY"
        if "dosage" in lower_query: return "NAVIGATE_TO_DOSAGE"
        if "contact" in lower_query: return "NAVIGATE_TO_CONTACT"
        if "home" in lower_query: return "NAVIGATE_TO_HOME"

    # If not a navigation command, proceed with RAG
    query_embedding = genai.embed_content(model=embedding_model, content=user_query, task_type="retrieval_query")['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=5) # Get more context
    
    if not results['documents'] or not results['documents'][0]:
        return "I do not have enough information to answer that."
        
    retrieved_context = "\n\n".join(results['documents'][0])

    # --- THIS IS THE NEW, ENHANCED PROMPT ---
    prompt = f"""
    You are "MediCare Plus AI Assistant", a friendly and helpful chatbot for a healthcare website.
    Your primary goal is to answer user questions by providing a concise summary based ONLY on the provided CONTEXT.
    After the summary, you MUST include a "Relevant Links:" section listing any relevant URLs or page references found in the context.

    **CRITICAL RULES:**
    1.  **Summarize First:** Read the user's QUESTION and provide a helpful, summary-style answer using only the information from the CONTEXT.
    2.  **List All Links:** After your summary, if the CONTEXT contains any URLs (like `https://...` or `safety.html`), list them under a "Relevant Links:" heading. Include download links for PDFs and links to other pages.
    3.  **Be Honest:** If the CONTEXT does not contain enough information to answer the QUESTION, you MUST respond with: "I do not have enough information from the website to answer that." Do not make up information.
    4.  **Stay in Character:** Be professional, helpful, and empathetic.

    ---
    CONTEXT:
    {retrieved_context}
    ---
    QUESTION:
    {user_query}
    ---
    ANSWER:
    """
    final_answer = chat_model.generate_content(prompt)
    return final_answer.text

# --- API Endpoints ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    bot_response = get_chatbot_response(user_message)
    return jsonify({"response": bot_response})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "database_initialized": collection is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
