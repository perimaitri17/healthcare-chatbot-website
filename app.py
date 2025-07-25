import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import re

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
    metadata_list = []
    
    for filename in os.listdir(html_folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(html_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                
                # Extract text content with better structure preservation
                page_content = soup.get_text(separator='\n', strip=True)
                
                # Extract links from the HTML
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text(strip=True)
                    if href and link_text:
                        links.append(f"{link_text}: {href}")
                
                # Create more meaningful chunks with better context
                lines = [line.strip() for line in page_content.split('\n') if len(line.strip()) > 20]
                
                # Group lines into larger, more contextual chunks
                chunk_size = 3  # Group 3 lines together for better context
                for i in range(0, len(lines), chunk_size):
                    chunk_lines = lines[i:i+chunk_size]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    # Add filename and links context to each chunk
                    enriched_chunk = f"Page: {filename}\n{chunk_text}"
                    if links:
                        enriched_chunk += f"\n\nAvailable links on this page:\n" + '\n'.join(links[:5])  # Limit to 5 links per chunk
                    
                    all_text_chunks.append(enriched_chunk)
                    metadata_list.append({
                        'filename': filename,
                        'chunk_index': len(all_text_chunks) - 1
                    })
    
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
        documents=all_text_chunks,
        metadatas=metadata_list
    )
    print(f"✅ Database initialized with {collection.count()} documents!")

initialize_database()

# --- Enhanced keyword matching function ---
def get_relevant_page_suggestions(user_query):
    """Return relevant page suggestions based on query keywords"""
    query_lower = user_query.lower()
    suggestions = []
    
    # Safety-related keywords
    safety_keywords = ['safety', 'safe', 'guideline', 'storage', 'administration', 'allergy', 'infection', 'emergency', 'procedure', 'protocol', 'risk', 'precaution']
    if any(keyword in query_lower for keyword in safety_keywords):
        suggestions.append("For detailed safety guidelines, visit: <a href='safety.html'>Safety Page</a>")
    
    # Dosage-related keywords
    dosage_keywords = ['dosage', 'dose', 'calculator', 'how much', 'amount', 'acetaminophen', 'ibuprofen', 'amoxicillin', 'dosing', 'medication amount']
    if any(keyword in query_lower for keyword in dosage_keywords):
        suggestions.append("For dosage calculations and guidelines, visit: <a href='dosage.html'>Dosage Page</a>")
    
    # Contact-related keywords
    contact_keywords = ['contact', 'phone', 'email', 'address', 'location', 'hours', 'reach', 'support', 'help', 'call']
    if any(keyword in query_lower for keyword in contact_keywords):
        suggestions.append("For contact information and support, visit: <a href='contact.html'>Contact Page</a>")
    
    return suggestions

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

    # Enhanced RAG with better retrieval
    query_embedding = genai.embed_content(model=embedding_model, content=user_query, task_type="retrieval_query")['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=8)  # Get more context
    
    if not results['documents'] or not results['documents'][0]:
        return "I do not have enough information to answer that."
        
    retrieved_context = "\n\n".join(results['documents'][0])

    # Get page suggestions based on keywords
    page_suggestions = get_relevant_page_suggestions(user_query)

    # --- ENHANCED PROMPT WITH BETTER INSTRUCTIONS ---
    prompt = f"""
You are "MediCare Plus AI Assistant", a friendly and helpful healthcare chatbot.

Your task is to:
1. Provide a helpful, detailed answer to the user's question using ONLY the information from the CONTEXT below
2. After your answer, include a "Relevant Links:" section with any links mentioned in the context
3. Include relevant page suggestions based on the user's query

**CRITICAL FORMATTING RULES:**
- Extract and display ALL links found in the context (both internal pages like safety.html and external URLs)
- Format internal page links as: <a href='filename.html'>Page Name</a>
- Include external URLs as complete clickable links
- If the context mentions downloadable files, include those links too
- Always include the page suggestions provided below

**CONTEXT:**
{retrieved_context}

**PAGE SUGGESTIONS:**
{' '.join(page_suggestions) if page_suggestions else 'For more information, visit our <a href="index.html">Home Page</a>'}

**USER QUESTION:**
{user_query}

**ANSWER FORMAT:**
[Your detailed answer here based on the context]

**Relevant Links:**
[List all links found in the context, plus the page suggestions above]

Remember: 
- Be professional and empathetic
- If the context doesn't contain enough information, say so honestly
- Always recommend consulting healthcare professionals for personalized advice
- Focus on providing accurate information from the context provided
"""

    try:
        final_answer = chat_model.generate_content(prompt)
        response_text = final_answer.text
        
        # Ensure we always have some relevant links
        if "Relevant Links:" not in response_text and page_suggestions:
            response_text += f"\n\n**Relevant Links:**\n" + '\n'.join(page_suggestions)
        
        return response_text
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team."

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
