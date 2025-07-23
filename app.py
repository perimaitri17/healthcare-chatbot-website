import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# This is a debug print statement to prove the latest code is running.
print("--- RUNNING LATEST CODE - GCP Cloud Run v1 ---")

# --- Basic Flask App Setup ---
app = Flask(__name__)

# Configure CORS specifically for your Netlify domain
CORS(app, origins=[
    'https://healthcare-chatbot-website.netlify.app',
    'http://localhost:3000',  # for local development
    'http://127.0.0.1:3000',  # alternative localhost
    'http://localhost:5000',  # if testing locally
], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type', 'Authorization'])

# Add manual CORS headers as backup
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://healthcare-chatbot-website.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# --- GLOBAL CONFIGURATION ---
# Securely get the API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=api_key)

embedding_model = 'models/embedding-001'
# CHANGED: Use gemini-1.5-flash for higher quotas
chat_model = genai.GenerativeModel('gemini-1.5-flash')
# The script will look for .html files in the main root directory
html_folder_path = '.'
collection_name = "healthcare_ai_docs"

# --- RATE LIMITING ---
import time
from collections import defaultdict

# Simple in-memory rate limiting
request_counts = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 10

def check_rate_limit(user_id="anonymous"):
    """Check if user has exceeded rate limit"""
    now = time.time()
    user_requests = request_counts[user_id]
    
    # Remove requests older than 1 minute
    user_requests[:] = [req_time for req_time in user_requests if now - req_time < 60]
    
    if len(user_requests) >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    user_requests.append(now)
    return True

# --- PERSISTENT DATABASE SETUP FOR GOOGLE CLOUD RUN ---
# Cloud Run has ephemeral filesystem, but we can use /tmp for temporary storage
# For production, you should use Google Cloud Storage or Firestore
possible_paths = [
    "/tmp/chromadb_data",  # Cloud Run writable temporary directory
    "./data",  # Local development
    "."  # Current directory as fallback
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
        
        # Process embeddings in smaller batches to avoid quota issues
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(all_text_chunks), batch_size):
            batch = all_text_chunks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_text_chunks)-1)//batch_size + 1}")
            
            try:
                response = genai.embed_content(
                    model=embedding_model, 
                    content=batch, 
                    task_type="retrieval_document"
                )
                all_embeddings.extend(response['embedding'])
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"Quota exceeded during embedding generation: {str(e)}")
                    return False
                raise e
        
        print(f"Generated {len(all_embeddings)} embeddings")

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
            embeddings=all_embeddings,
            documents=all_text_chunks
        )
        
        # Verify the collection was created successfully
        doc_count = collection.count()
        print(f"✅ New database initialized successfully with {doc_count} documents!")
        return True
        
    except Exception as e:
        print(f"ERROR during database initialization: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            print("This appears to be a quota/rate limit issue.")
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
You are a helpful healthcare AI assistant. Answer the user's question based ONLY on the following context.
If the context doesn't contain the answer, say "I do not have enough information to answer that."
Always recommend consulting healthcare professionals for medical advice.

CONTEXT:
{retrieved_context}

QUESTION:
{user_query}
"""
        final_answer = chat_model.generate_content(prompt)
        return final_answer.text
        
    except Exception as e:
        print(f"ERROR in get_chatbot_response: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            return "I'm currently experiencing high usage. Please try again in a few minutes."
        return f"Sorry, I encountered an error processing your request. Please try again."

# --- API Endpoints ---
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        return response
    
    # Rate limiting
    user_id = request.json.get('userId', 'anonymous') if request.json else 'anonymous'
    if not check_rate_limit(user_id):
        return jsonify({
            "error": "Too many requests. Please wait a minute before sending another message.",
            "retryAfter": 60
        }), 429
    
    if not database_ready:
        return jsonify({"error": "Database not ready. Please check server logs."}), 500
        
    user_message = request.json.get('message') if request.json else None
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Validate message length    
    if len(user_message) > 2000:
        return jsonify({"error": "Message too long. Please keep it under 2000 characters."}), 400
        
    try:
        bot_response = get_chatbot_response(user_message)
        return jsonify({
            "response": bot_response,
            "model": "gemini-1.5-flash",
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            return jsonify({
                "error": "I'm currently experiencing high usage. Please try again in a few minutes.",
                "quotaExceeded": True
            }), 429
        return jsonify({"error": "Internal server error"}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "database_ready": database_ready,
        "collection_exists": collection is not None,
        "document_count": collection.count() if collection else 0,
        "server_status": "running",
        "platform": "Google Cloud Run"
    }
    return jsonify(status)

# Add a simple test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Server is working!", 
        "cors": "enabled",
        "platform": "Google Cloud Run"
    })

# --- Run the App ---
if __name__ == '__main__':
    if database_ready:
        print("🚀 Server starting with database ready!")
    else:
        print("⚠️  Server starting but database is NOT ready!")
    
    print("CORS configured for: https://healthcare-chatbot-website.netlify.app")
    
    # CHANGED: Use port from environment variable (Cloud Run requirement)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
