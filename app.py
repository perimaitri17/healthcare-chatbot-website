import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# This is a debug print statement to prove the latest code is running.
print("--- RUNNING LATEST CODE - v4 ---")

# --- Basic Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- GLOBAL CONFIGURATION ---
# Securely get the API key from an environment variable on Render
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=api_key)

embedding_model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-pro-latest')
# The script will look for .html files in the main root directory
html_folder_path = '.'
collection_name = "healthcare_ai_docs"

# --- PERSISTENT DATABASE SETUP ---
# Try multiple possible paths for Render's persistent disk
possible_paths = [
    os.getenv("RENDER_DATA_PATH", "/opt/render/project/src/data"),  # Render-specific
    "./data",  # Relative path in project
    "/tmp/chromadb_data",  # Temporary fallback
    "."  # Current directory as last resort
]

db_path = None
for path in possible_paths:
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(path, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        db_path = path
        print(f"✅ Using database path: {db_path}")
        break
    except Exception as e:
        print(f"❌ Cannot use path {path}: {str(e)}")
        continue

if not db_path:
    raise RuntimeError("No writable path found for ChromaDB database!")

client = chromadb.PersistentClient(path=db_path)
collection = None

# --- Function to load, chunk, and create the database ---
def initialize_database():
    global collection
    print("Initializing new database...")

    # First, make sure the directory exists and list what's in it
    print(f"Contents of {html_folder_path}: {os.listdir(html_folder_path)}")
    
    all_text_chunks = []
    html_files_found = []
    
    print(f"Reading HTML files from: {html_folder_path}")
    
    # Check if directory exists
    if not os.path.exists(html_folder_path):
        print(f"ERROR: Directory {html_folder_path} does not exist!")
        return False
    
    # Process HTML files
    for filename in os.listdir(html_folder_path):
        if filename.endswith(".html"):
            html_files_found.append(filename)
            file_path = os.path.join(html_folder_path, filename)
            print(f"Processing file: {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'lxml')
                    text = soup.get_text(separator='\n', strip=True)
                    chunks = [chunk.strip() for chunk in text.split('\n') if len(chunk.strip()) > 10]
                    all_text_chunks.extend(chunks)
                    print(f"  Added {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"ERROR processing {filename}: {str(e)}")
    
    print(f"HTML files found: {html_files_found}")
    print(f"Total text chunks collected: {len(all_text_chunks)}")

    if not all_text_chunks:
        print("ERROR: No text chunks found. Cannot initialize database.")
        print("This could be because:")
        print("1. No .html files found in the directory")
        print("2. HTML files are empty or contain only short text")
        print("3. File reading permissions issue")
        return False

    try:
        print("Generating embeddings with Gemini AI...")
        response = genai.embed_content(
            model=embedding_model, 
            content=all_text_chunks, 
            task_type="retrieval_document"
        )
        embeddings = response['embedding']
        print(f"Generated {len(embeddings)} embeddings")

        # Delete existing collection if it exists
        try:
            existing_collection = client.get_collection(collection_name)
            client.delete_collection(collection_name)
            print("Deleted existing collection")
        except:
            pass  # Collection doesn't exist, which is fine

        # Create new collection
        collection = client.create_collection(collection_name)
        collection.add(
            ids=[f"doc_{i}" for i in range(len(all_text_chunks))],
            embeddings=embeddings,
            documents=all_text_chunks
        )
        
        # Verify the collection was created successfully
        doc_count = collection.count()
        print(f"✅ New database initialized successfully with {doc_count} documents!")
        return True
        
    except Exception as e:
        print(f"ERROR during database initialization: {str(e)}")
        return False

# --- Check for Database on Startup ---
database_ready = False

try:
    # Try to get existing collection
    collection = client.get_collection(collection_name)
    doc_count = collection.count()
    print(f"✅ Connected to existing database with {doc_count} documents.")
    database_ready = True
except Exception as e:
    print(f"Database collection '{collection_name}' not found or error occurred: {str(e)}")
    print("Creating a new database...")
    
    success = initialize_database()
    if success:
        database_ready = True
    else:
        print("CRITICAL ERROR: Failed to initialize database!")

# --- Main Chatbot Logic Function ---
def get_chatbot_response(user_query):
    global collection, database_ready
    
    if not database_ready or not collection:
        print("CRITICAL ERROR: get_chatbot_response called but database is not ready.")
        return "Sorry, the chatbot database is not available. Please check the server logs and ensure HTML files are present."

    try:
        # Generate embedding for the query
        query_embedding = genai.embed_content(
            model=embedding_model, 
            content=user_query, 
            task_type="retrieval_query"
        )['embedding']
        
        # Query the collection
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        
        if not results['documents'][0]:
            return "I do not have enough information to answer that."
        
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
        
    except Exception as e:
        print(f"ERROR in get_chatbot_response: {str(e)}")
        return f"Sorry, I encountered an error processing your request: {str(e)}"

# --- API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    if not database_ready:
        return jsonify({"error": "Database not ready. Please check server logs."}), 500
        
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
        
    bot_response = get_chatbot_response(user_message)
    return jsonify({"response": bot_response})

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "database_ready": database_ready,
        "collection_exists": collection is not None,
        "document_count": collection.count() if collection else 0
    }
    return jsonify(status)

# --- Run the App ---
if __name__ == '__main__':
    if database_ready:
        print("🚀 Server starting with database ready!")
    else:
        print("⚠️  Server starting but database is NOT ready!")
    
    app.run(host='0.0.0.0', port=5000)
