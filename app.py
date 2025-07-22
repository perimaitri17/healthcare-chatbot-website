import chromadb
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS

# --- Basic Flask App Setup ---
app = Flask(__name__)
CORS(app) # This enables Cross-Origin Resource Sharing

# --- CONFIGURATION (Same as before) ---
try:
    genai.configure(api_key="AIzaSyBDBs9WhF46leK7tDfZA_UqznXjr-c0_Kg")
    embedding_model = 'models/embedding-001'
    chat_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Connect to ChromaDB
    client = chromadb.Client()
    collection = client.get_collection("healthcare_ai_docs")
    print("✅ ChromaDB and Gemini configured successfully.")

except Exception as e:
    print(f"❌ Error during configuration: {e}")


# --- Main Chatbot Logic in a Function ---
def get_chatbot_response(user_query):
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=user_query,
        task_type="retrieval_query"
    )['embedding']

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
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


# --- API Endpoint Definition ---
@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's message from the incoming request
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get the response from our chatbot logic
    bot_response = get_chatbot_response(user_message)
    
    # Return the response as JSON
    return jsonify({"response": bot_response})

# This allows you to run the app by typing "py app.py"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)