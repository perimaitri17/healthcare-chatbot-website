import os
import chromadb
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

print("--- GCP Chatbot Service Starting [TARGETED RESPONSES] ---")

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

# --- KEYWORD-BASED RESPONSE MAPPING ---
def get_targeted_links(query_lower):
    """Return only relevant links based on keywords"""
    links = {}
    
    # Safety-related keywords
    if any(word in query_lower for word in ['safety', 'safe', 'storage', 'administration', 'emergency', 'allergy', 'infection', 'risk', 'precaution', 'guideline', 'ppe', 'sterile', 'hygiene']):
        links['safety'] = "https://healthcare-chatbot-website.netlify.app/safety.html"
    
    # Dosage-related keywords
    if any(word in query_lower for word in ['dosage', 'dose', 'amount', 'how much', 'calculator', 'acetaminophen', 'ibuprofen', 'amoxicillin', 'medication', 'mg', 'ml', 'prescription']):
        links['dosage'] = "https://healthcare-chatbot-website.netlify.app/dosage.html"
    
    # Contact-related keywords
    if any(word in query_lower for word in ['contact', 'phone', 'email', 'address', 'support', 'help', 'call', 'reach', 'emergency']):
        links['contact'] = "https://healthcare-chatbot-website.netlify.app/contact.html"
    
    # Download-related keywords
    if 'download' in query_lower:
        if 'pdf' in query_lower or len([word for word in ['download'] if word in query_lower]) == 1:
            # If only "download" is mentioned or "pdf" is specified, show all PDFs
            links['downloads'] = [
                "https://www.africau.edu/images/default/sample.pdf",
                "https://www.w3.org/WAI/ER/PRACTICES/pdf/text-document.pdf", 
                "https://www.ets.org/s/gre/pdf/gre_info_test_centers.pdf"
            ]
    
    # External site keywords
    if 'who' in query_lower:
        links['who'] = "https://www.who.int/"
    if 'cdc' in query_lower:
        links['cdc'] = "https://www.cdc.gov/"
    if 'nih' in query_lower:
        links['nih'] = "https://www.nih.gov/"
    
    return links

def handle_navigation(user_query):
    """Handle navigation commands and return appropriate response"""
    lower_query = user_query.lower()
    
    # Check for navigation commands
    if any(nav_word in lower_query for nav_word in ["go to", "take me to", "navigate to", "navigate"]):
        if "safety" in lower_query:
            return "NAVIGATE:safety"
        elif "dosage" in lower_query:
            return "NAVIGATE:dosage"
        elif "contact" in lower_query:
            return "NAVIGATE:contact"
        elif "home" in lower_query:
            return "NAVIGATE:home"
    
    return None

def format_targeted_response(content, links, query):
    """Format response with only relevant links"""
    response_parts = [content.strip()]
    
    if links:
        response_parts.append("\n**Relevant Links:**")
        
        # Format different types of links
        if 'safety' in links:
            response_parts.append(f"• Safety Guidelines: {links['safety']}")
        
        if 'dosage' in links:
            response_parts.append(f"• Dosage Calculator: {links['dosage']}")
        
        if 'contact' in links:
            response_parts.append(f"• Contact Us: {links['contact']}")
        
        if 'downloads' in links:
            if len(links['downloads']) == 1:
                response_parts.append(f"• Download PDF: {links['downloads'][0]}")
            else:
                response_parts.append("• Download PDFs:")
                for pdf_link in links['downloads']:
                    response_parts.append(f"  - {pdf_link}")
        
        if 'who' in links:
            response_parts.append(f"• WHO Site: {links['who']}")
        
        if 'cdc' in links:
            response_parts.append(f"• CDC Site: {links['cdc']}")
        
        if 'nih' in links:
            response_parts.append(f"• NIH Site: {links['nih']}")
    
    # Add home link only if no specific links were found
    if not links:
        response_parts.append(f"\n• Home: https://healthcare-chatbot-website.netlify.app/index.html")
    
    return "\n".join(response_parts)

# --- ENHANCED Response Function ---
def get_chatbot_response(user_query):
    global collection
    if not collection:
        return "Sorry, the chatbot database is not initialized."

    print(f"Processing query: '{user_query}'")
    
    # Handle navigation first
    nav_response = handle_navigation(user_query)
    if nav_response:
        page = nav_response.split(':')[1]
        return f"NAVIGATE_TO_{page.upper()}"
    
    lower_query = user_query.lower()
    
    # Get targeted links based on keywords
    relevant_links = get_targeted_links(lower_query)
    
    # Enhanced vector search
    query_embedding = genai.embed_content(model=embedding_model, content=user_query, task_type="retrieval_query")['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=3)  # Reduced to 3 for more focused results
    
    if not results['documents'] or not results['documents'][0]:
        return generate_fallback_response(user_query, relevant_links)
        
    retrieved_context = "\n\n".join(results['documents'][0])
    
    # FOCUSED PROMPT - Only relevant information
    prompt = f"""
You are MediCare Plus AI Assistant. Answer the user's question concisely using ONLY the provided context.

CONTEXT:
{retrieved_context}

USER QUESTION: {user_query}

RESPONSE RULES:
1. Give a brief, focused answer (2-3 sentences max)
2. Do NOT include any links in your response - links will be added separately
3. Focus only on answering the specific question asked
4. Be concise and direct

ANSWER:
"""

    try:
        final_answer = chat_model.generate_content(prompt)
        response_content = final_answer.text
        
        # Format with only relevant links
        formatted_response = format_targeted_response(response_content, relevant_links, user_query)
        formatted_response += "\n\nFor personalized medical advice, consult your healthcare provider."
        
        print(f"Generated targeted response length: {len(formatted_response)}")
        return formatted_response
        
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return generate_fallback_response(user_query, relevant_links)

def generate_fallback_response(user_query, relevant_links):
    """Generate a focused fallback response"""
    query_lower = user_query.lower()
    
    if "safety" in query_lower:
        content = "Key medication safety practices include proper storage, checking expiration dates, verifying patient identity, and maintaining sterile environments when necessary."
    elif any(word in query_lower for word in ['dosage', 'dose', 'amount']):
        content = "Proper medication dosing requires following prescriber instructions, using appropriate measuring devices, and considering patient factors like age and weight."
    elif any(word in query_lower for word in ['contact', 'phone', 'email', 'support']):
        content = "Our healthcare support team is available to assist you with questions and provide guidance on medication safety and dosing."
    else:
        content = "I'm here to help with healthcare questions, medication safety, and dosage information."
    
    return format_targeted_response(content, relevant_links, user_query) + "\n\nFor personalized medical advice, consult your healthcare provider."

# --- API Endpoints ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    bot_response = get_chatbot_response(user_message)
    
    # Handle navigation responses
    if bot_response.startswith("NAVIGATE_TO_"):
        page = bot_response.split("NAVIGATE_TO_")[1].lower()
        return jsonify({
            "response": f"Taking you to the {page} page...",
            "navigate": f"https://healthcare-chatbot-website.netlify.app/{page}.html" if page != "home" else "https://healthcare-chatbot-website.netlify.app/index.html"
        })
    
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
