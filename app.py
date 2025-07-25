import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import re

print("--- GCP Chatbot Service Starting [DIAGNOSTIC VERSION] ---")

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
    print("=== DIAGNOSTIC: Initializing in-memory vector database ===")
    
    # First, let's see what files we have
    print(f"Looking for HTML files in: {html_folder_path}")
    all_files = os.listdir(html_folder_path)
    html_files = [f for f in all_files if f.endswith(".html")]
    print(f"All files in directory: {all_files}")
    print(f"HTML files found: {html_files}")
    
    all_text_chunks = []
    metadata_list = []
    
    for filename in html_files:
        print(f"\n--- Processing file: {filename} ---")
        file_path = os.path.join(html_folder_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                print(f"File size: {len(html_content)} characters")
                
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Extract text content
                page_content = soup.get_text(separator='\n', strip=True)
                print(f"Extracted text length: {len(page_content)} characters")
                print(f"First 200 chars: {page_content[:200]}...")
                
                # Extract links from the HTML
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text(strip=True)
                    if href and link_text:
                        links.append(f"{link_text}: {href}")
                
                print(f"Links found: {len(links)}")
                for link in links[:5]:  # Show first 5 links
                    print(f"  - {link}")
                
                # Create chunks
                lines = [line.strip() for line in page_content.split('\n') if len(line.strip()) > 10]
                print(f"Valid lines found: {len(lines)}")
                
                # Group lines into chunks
                chunk_size = 5  # Larger chunks for better context
                chunk_count = 0
                for i in range(0, len(lines), chunk_size):
                    chunk_lines = lines[i:i+chunk_size]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    # Add filename and links context to each chunk
                    enriched_chunk = f"=== PAGE: {filename} ===\n{chunk_text}"
                    if links:
                        enriched_chunk += f"\n\n=== AVAILABLE LINKS ===\n" + '\n'.join(links)
                    
                    all_text_chunks.append(enriched_chunk)
                    metadata_list.append({
                        'filename': filename,
                        'chunk_index': len(all_text_chunks) - 1,
                        'has_links': len(links) > 0
                    })
                    chunk_count += 1
                
                print(f"Created {chunk_count} chunks for {filename}")
                
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
    
    print(f"\n=== TOTAL CHUNKS CREATED: {len(all_text_chunks)} ===")
    
    if not all_text_chunks:
        print("CRITICAL: No text chunks found.")
        return

    print("Generating embeddings...")
    try:
        response = genai.embed_content(model=embedding_model, content=all_text_chunks, task_type="retrieval_document")
        embeddings = response['embedding']
        print(f"Embeddings generated: {len(embeddings)} vectors")
        
        collection = client.create_collection(collection_name)
        collection.add(
            ids=[f"doc_{i}" for i in range(len(all_text_chunks))],
            embeddings=embeddings,
            documents=all_text_chunks,
            metadatas=metadata_list
        )
        print(f"✅ Database initialized with {collection.count()} documents!")
        
        # Test query to see if data is accessible
        test_query = "safety"
        test_embedding = genai.embed_content(model=embedding_model, content=test_query, task_type="retrieval_query")['embedding']
        test_results = collection.query(query_embeddings=[test_embedding], n_results=3)
        print(f"\n=== TEST QUERY RESULTS for '{test_query}' ===")
        if test_results['documents'] and test_results['documents'][0]:
            for i, doc in enumerate(test_results['documents'][0]):
                print(f"Result {i+1}: {doc[:200]}...")
        else:
            print("NO RESULTS FOUND FOR TEST QUERY!")
            
    except Exception as e:
        print(f"ERROR during embedding/database creation: {e}")

initialize_database()

# --- Enhanced keyword matching function ---
def get_relevant_page_suggestions(user_query):
    """Return relevant page suggestions based on query keywords"""
    query_lower = user_query.lower()
    suggestions = []
    
    print(f"DIAGNOSTIC: Analyzing query for keywords: '{query_lower}'")
    
    # Safety-related keywords
    safety_keywords = ['safety', 'safe', 'guideline', 'storage', 'administration', 'allergy', 'infection', 'emergency', 'procedure', 'protocol', 'risk', 'precaution']
    safety_matches = [kw for kw in safety_keywords if kw in query_lower]
    if safety_matches:
        print(f"DIAGNOSTIC: Safety keywords matched: {safety_matches}")
        suggestions.append("For detailed safety guidelines, visit: <a href='safety.html'>Safety Page</a>")
    
    # Dosage-related keywords
    dosage_keywords = ['dosage', 'dose', 'calculator', 'how much', 'amount', 'acetaminophen', 'ibuprofen', 'amoxicillin', 'dosing', 'medication amount']
    dosage_matches = [kw for kw in dosage_keywords if kw in query_lower]
    if dosage_matches:
        print(f"DIAGNOSTIC: Dosage keywords matched: {dosage_matches}")
        suggestions.append("For dosage calculations and guidelines, visit: <a href='dosage.html'>Dosage Page</a>")
    
    # Contact-related keywords
    contact_keywords = ['contact', 'phone', 'email', 'address', 'location', 'hours', 'reach', 'support', 'help', 'call']
    contact_matches = [kw for kw in contact_keywords if kw in query_lower]
    if contact_matches:
        print(f"DIAGNOSTIC: Contact keywords matched: {contact_matches}")
        suggestions.append("For contact information and support, visit: <a href='contact.html'>Contact Page</a>")
    
    print(f"DIAGNOSTIC: Generated {len(suggestions)} page suggestions")
    return suggestions

# --- Main Chatbot Logic Function ---
def get_chatbot_response(user_query):
    global collection
    print(f"\n=== PROCESSING QUERY: '{user_query}' ===")
    
    if not collection:
        print("ERROR: Database not initialized")
        return "Sorry, the chatbot database is not initialized."

    # First, check for simple navigation commands
    lower_query = user_query.lower()
    print(f"DIAGNOSTIC: Checking for navigation commands in: '{lower_query}'")
    
    if "go to" in lower_query or "take me to" in lower_query or "navigate to" in lower_query:
        print("DIAGNOSTIC: Navigation command detected")
        if "safety" in lower_query: 
            print("DIAGNOSTIC: Navigating to safety")
            return "I'll take you to the safety page now. NAVIGATE_TO_SAFETY"
        if "dosage" in lower_query: 
            print("DIAGNOSTIC: Navigating to dosage")
            return "I'll take you to the dosage page now. NAVIGATE_TO_DOSAGE"
        if "contact" in lower_query: 
            print("DIAGNOSTIC: Navigating to contact")
            return "I'll take you to the contact page now. NAVIGATE_TO_CONTACT"
        if "home" in lower_query: 
            print("DIAGNOSTIC: Navigating to home")
            return "I'll take you to the home page now. NAVIGATE_TO_HOME"

    # Enhanced RAG with better retrieval
    print("DIAGNOSTIC: Performing vector search...")
    try:
        query_embedding = genai.embed_content(model=embedding_model, content=user_query, task_type="retrieval_query")['embedding']
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        
        print(f"DIAGNOSTIC: Vector search returned {len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0} results")
        
        if not results['documents'] or not results['documents'][0]:
            print("DIAGNOSTIC: No documents found in vector search")
            return "I do not have enough information to answer that."
            
        retrieved_context = "\n\n".join(results['documents'][0])
        print(f"DIAGNOSTIC: Retrieved context length: {len(retrieved_context)} characters")
        print(f"DIAGNOSTIC: Context preview: {retrieved_context[:300]}...")
        
        # Get page suggestions based on keywords
        page_suggestions = get_relevant_page_suggestions(user_query)

        # Simple response generation for testing
        if "safety" in user_query.lower():
            response = f"""Based on our safety guidelines:

{retrieved_context[:500]}...

**Relevant Links:**
- <a href='safety.html'>Safety Page</a> - Complete safety guidelines and protocols
- <a href='index.html'>Home Page</a> - Main healthcare information

{' | '.join(page_suggestions) if page_suggestions else ''}

Please consult with healthcare professionals for personalized safety advice."""
            
            return response
        
        # For other queries, use the AI model
        prompt = f"""
You are MediCare Plus AI Assistant. Answer the user's question using the context provided.

CONTEXT:
{retrieved_context}

USER QUESTION: {user_query}

INSTRUCTIONS:
1. Provide a helpful answer based on the context
2. Include any links mentioned in the context
3. Add relevant page suggestions: {' | '.join(page_suggestions) if page_suggestions else 'Visit our main pages for more information'}

Answer:"""

        final_answer = chat_model.generate_content(prompt)
        response_text = final_answer.text
        
        # Ensure we always have some relevant links
        if "href=" not in response_text and page_suggestions:
            response_text += f"\n\n**Relevant Links:**\n" + '\n'.join(page_suggestions)
        
        print(f"DIAGNOSTIC: Generated response length: {len(response_text)} characters")
        return response_text
        
    except Exception as e:
        print(f"ERROR generating response: {e}")
        return f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"

# --- API Endpoints ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    print(f"\n=== API CALL RECEIVED ===")
    print(f"Message: {user_message}")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        bot_response = get_chatbot_response(user_message)
        print(f"=== API RESPONSE ===")
        print(f"Response length: {len(bot_response)} characters")
        print(f"Response preview: {bot_response[:200]}...")
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"ERROR in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    health_info = {
        "status": "ok", 
        "database_initialized": collection is not None,
        "collection_count": collection.count() if collection else 0
    }
    print(f"Health check: {health_info}")
    return jsonify(health_info)

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check what's in the database"""
    if not collection:
        return jsonify({"error": "Database not initialized"})
    
    # Get a sample of documents
    sample_results = collection.query(query_embeddings=[[0.1] * 768], n_results=3)  # Dummy query
    
    debug_data = {
        "total_documents": collection.count(),
        "sample_documents": sample_results['documents'][0] if sample_results['documents'] else [],
        "sample_metadata": sample_results['metadatas'][0] if sample_results['metadatas'] else []
    }
    
    return jsonify(debug_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
