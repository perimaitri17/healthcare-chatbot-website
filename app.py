import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

print("--- GCP Chatbot Service Starting [TARGETED FIX] ---")

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
    print("Reading HTML files from:", html_folder_path)
    all_text_chunks = []
    
    for filename in os.listdir(html_folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(html_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                
                # Extract all text content
                page_content = soup.get_text(separator='\n', strip=True)
                
                # Extract all links
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text(strip=True)
                    if href and link_text:
                        links.append(f"Link: {link_text} -> {href}")
                
                # Create larger, more contextual chunks
                lines = [line.strip() for line in page_content.split('\n') if len(line.strip()) > 15]
                
                # Group lines into larger chunks (10 lines each)
                chunk_size = 10
                for i in range(0, len(lines), chunk_size):
                    chunk_lines = lines[i:i+chunk_size]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    # Create rich context for each chunk
                    enriched_chunk = f"""
PAGE: {filename}
CONTENT:
{chunk_text}

AVAILABLE_LINKS:
{chr(10).join(links) if links else 'No links available'}

KEYWORDS: {filename.replace('.html', '')} healthcare medical safety dosage contact
"""
                    all_text_chunks.append(enriched_chunk)
    
    print(f"Found {len(all_text_chunks)} total text chunks.")
    
    if not all_text_chunks:
        print("CRITICAL: No text chunks found.")
        return

    print("Generating embeddings with Gemini AI...")
    response = genai.embed_content(model=embedding_model, content=all_text_chunks, task_type="retrieval_document")
    embeddings = response['embedding']
    
    collection = client.create_collection(collection_name)
    collection.add(
        ids=[f"doc_{i}" for i in range(len(all_text_chunks))],
        embeddings=embeddings,
        documents=all_text_chunks
    )
    print(f"✅ In-memory database initialized successfully with {collection.count()} documents!")

initialize_database()

# --- ENHANCED Response Function ---
def get_chatbot_response(user_query):
    global collection
    if not collection:
        return "Sorry, the chatbot database is not initialized."

    print(f"Processing query: '{user_query}'")
    
    # Enhanced navigation commands
    lower_query = user_query.lower()
    if any(nav_word in lower_query for nav_word in ["go to", "take me to", "navigate to", "navigate"]):
        if "safety" in lower_query: 
            return "I'll take you to our comprehensive safety guidelines page now. NAVIGATE_TO_SAFETY"
        if "dosage" in lower_query: 
            return "I'll take you to our dosage calculator and guidelines page now. NAVIGATE_TO_DOSAGE"
        if "contact" in lower_query: 
            return "I'll take you to our contact information page now. NAVIGATE_TO_CONTACT"
        if "home" in lower_query: 
            return "I'll take you to our home page now. NAVIGATE_TO_HOME"

    # Enhanced vector search
    query_embedding = genai.embed_content(model=embedding_model, content=user_query, task_type="retrieval_query")['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=6)
    
    if not results['documents'] or not results['documents'][0]:
        return generate_fallback_response(user_query)
        
    retrieved_context = "\n\n".join(results['documents'][0])
    print(f"Retrieved context length: {len(retrieved_context)}")

    # Generate smart page suggestions
    page_suggestions = get_smart_suggestions(user_query)
    
    # SIMPLIFIED BUT EFFECTIVE PROMPT
    prompt = f"""
You are MediCare Plus AI Assistant, a helpful healthcare chatbot.

TASK: Answer the user's question using the provided CONTEXT. Always include relevant links.

CONTEXT:
{retrieved_context}

USER QUESTION: {user_query}

RESPONSE RULES:
1. Give a helpful answer based on the CONTEXT
2. Extract and include ALL links found in the context
3. Format links as: <a href='filename.html'>Page Name</a> or <a href='full-url'>Link Text</a>
4. Add these relevant page suggestions: {page_suggestions}
5. Always end with "For personalized medical advice, consult your healthcare provider."

ANSWER:
"""

    try:
        final_answer = chat_model.generate_content(prompt)
        response = final_answer.text
        
        # Ensure links are included
        if not any(indicator in response for indicator in ["<a href", "http", "safety.html", "dosage.html", "contact.html"]):
            response += f"\n\n**Quick Links:**\n{page_suggestions}"
        
        print(f"Generated response length: {len(response)}")
        return response
        
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return generate_fallback_response(user_query)

def get_smart_suggestions(user_query):
    """Generate smart page suggestions based on query content"""
    query_lower = user_query.lower()
    suggestions = []
    
    # Safety-related
    if any(word in query_lower for word in ['safety', 'safe', 'storage', 'administration', 'emergency', 'allergy', 'infection', 'risk', 'precaution', 'guideline']):
        suggestions.append("• <a href='safety.html'>Safety Guidelines</a> - Comprehensive medication safety protocols")
    
    # Dosage-related  
    if any(word in query_lower for word in ['dosage', 'dose', 'amount', 'how much', 'calculator', 'acetaminophen', 'ibuprofen', 'amoxicillin', 'medication']):
        suggestions.append("• <a href='dosage.html'>Dosage Calculator</a> - Interactive medication dosing tools")
    
    # Contact-related
    if any(word in query_lower for word in ['contact', 'phone', 'email', 'address', 'support', 'help', 'call', 'reach']):
        suggestions.append("• <a href='contact.html'>Contact Us</a> - Get in touch with our support team")
    
    # Always include home page
    suggestions.append("• <a href='index.html'>Home</a> - Main healthcare information hub")
    
    return "\n".join(suggestions)

def generate_fallback_response(user_query):
    """Generate a helpful fallback response when vector search fails"""
    query_lower = user_query.lower()
    
    if "safety" in query_lower:
        return """
**Medication Safety Guidelines:**

Key safety practices include:
- Store medications in cool, dry places away from sunlight
- Keep medications in original containers with labels
- Check expiration dates regularly
- Never share prescription medications
- Keep medications away from children and pets

**Relevant Resources:**
• <a href='safety.html'>Complete Safety Guidelines</a>
• <a href='contact.html'>Contact Support</a>
• <a href='https://www.fda.gov/drugs/information-consumers-and-patients-drugs/'>FDA Drug Safety</a>

For personalized medical advice, consult your healthcare provider.
"""
    
    elif any(word in query_lower for word in ['dosage', 'dose', 'amount']):
        return """
**Medication Dosage Information:**

Proper dosing is crucial for medication effectiveness and safety:
- Always follow prescriber instructions
- Use appropriate measuring devices
- Consider patient age, weight, and medical conditions
- Never exceed recommended doses

**Dosage Resources:**
• <a href='dosage.html'>Dosage Calculator</a>
• <a href='safety.html'>Safety Guidelines</a>
• <a href='contact.html'>Contact Support</a>

For personalized medical advice, consult your healthcare provider.
"""
    
    elif any(word in query_lower for word in ['contact', 'phone', 'email', 'support']):
        return """
**Contact Information:**

Get in touch with our healthcare support team:
- Phone: +91 11 2345 6789
- Email: info@medicareplus.com
- Address: 123 Healthcare Street, New Delhi 110001

**Quick Links:**
• <a href='contact.html'>Full Contact Details</a>
• <a href='index.html'>Home Page</a>
• <a href='safety.html'>Safety Guidelines</a>

For personalized medical advice, consult your healthcare provider.
"""
    
    else:
        return """
I'm here to help with healthcare questions, medication safety, dosage calculations, and more.

**Popular Resources:**
• <a href='safety.html'>Safety Guidelines</a> - Medication safety protocols
• <a href='dosage.html'>Dosage Calculator</a> - Interactive dosing tools  
• <a href='contact.html'>Contact Us</a> - Support and information
• <a href='index.html'>Home</a> - Main healthcare hub

How can I assist you with your healthcare needs today?

For personalized medical advice, consult your healthcare provider.
"""

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
    return jsonify({
        "status": "ok", 
        "database_initialized": collection is not None,
        "total_documents": collection.count() if collection else 0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
