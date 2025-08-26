import streamlit as st
import openai
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import numpy as np
import time
import json
from datetime import datetime
import os
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
K_RETRIEVAL = 3  # Number of relevant chunks to retrieve
MAX_TOKENS = 500

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-3.5-turbo"
if "knowledge_base_loaded" not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if "knowledge_content" not in st.session_state:
    st.session_state.knowledge_content = ""

# Knowledge base file path (should be in the same directory as the script)
KNOWLEDGE_BASE_FILE = "Paystack_FAQ_and_Documentation.txt"

class CustomerSupportBot:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_knowledge_base(self, file_path):
        """Load knowledge base from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            st.error(f"Knowledge base file not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            return None
        
    def process_knowledge_base(self, text, openai_api_key):
        """Process and vectorize the knowledge base"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Create vector store
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            return vectorstore
        except Exception as e:
            st.error(f"Error processing knowledge base: {str(e)}")
            return None
    
    def retrieve_relevant_context(self, query, vectorstore, k=K_RETRIEVAL):
        """Retrieve relevant context from vector store"""
        try:
            docs = vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return ""
    
    def generate_response_openai(self, query, context, api_key):
        """Generate response using OpenAI GPT-3.5-turbo"""
        try:
            openai.api_key = api_key
            
            prompt = f"""You are a helpful Paystack customer support agent. Use the following context to answer the user's question accurately and helpfully about Paystack's digital payment services in Nigeria.

IMPORTANT GUARDRAILS:
- If the context doesn't contain relevant information, politely explain that you need to escalate to a human agent
- Always recommend contacting Paystack support if you're unsure about any process
- For sensitive account issues, direct users to contact support@paystack.com or +234 1 700 5000
- Never provide financial advice beyond what's in the documentation
- Always emphasize security best practices for payment transactions
- If asked about competitor services, redirect to Paystack's features

Context:
{context}

User Question: {query}

Please provide a helpful, professional response focused on Paystack services. If you cannot answer based on the context provided, suggest contacting Paystack support at support@paystack.com or +234 1 700 5000."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties. Please contact Paystack support at support@paystack.com or +234 1 700 5000. Error: {str(e)}"
    
    def generate_response_gemini(self, query, context, api_key):
        """Generate response using Gemini-1.5-flash"""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""You are a helpful Paystack customer support agent. Use the following context to answer the user's question accurately and helpfully about Paystack's digital payment services in Nigeria.

IMPORTANT GUARDRAILS:
- If the context doesn't contain relevant information, politely explain that you need to escalate to a human agent
- Always recommend contacting Paystack support if you're unsure about any process
- For sensitive account issues, direct users to contact support@paystack.com or +234 1 700 5000
- Never provide financial advice beyond what's in the documentation
- Always emphasize security best practices for payment transactions
- If asked about competitor services, redirect to Paystack's features

Context:
{context}

User Question: {query}

Please provide a helpful, professional response focused on Paystack services. If you cannot answer based on the context provided, suggest contacting Paystack support at support@paystack.com or +234 1 700 5000."""

            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties. Please contact Paystack support at support@paystack.com or +234 1 700 5000. Error: {str(e)}"

def setup_sidebar():
    """Setup sidebar with configuration options"""
    st.sidebar.title("AI Support Configuration")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select AI Model",
        ["GPT-3.5-turbo", "Gemini-1.5-flash"],
        index=0
    )
    st.session_state.selected_model = model_option
    
    # Auto-load knowledge base on first run
    if not st.session_state.knowledge_base_loaded:
        bot = CustomerSupportBot()
        content = bot.load_knowledge_base(KNOWLEDGE_BASE_FILE)
    
    
    if st.session_state.knowledge_base_loaded and "GPT" in model_option and st.session_state.vectorstore is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            with st.spinner("Processing knowledge base for RAG..."):
                bot = CustomerSupportBot()
                vectorstore = bot.process_knowledge_base(st.session_state.knowledge_content, openai_api_key)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.sidebar.success("✅ Knowledge base processed and vectorized!")
                else:
                    st.sidebar.error("❌ Failed to process knowledge base")
        else:
            st.sidebar.error("OpenAI API key not found in environment variables")
    

    
    # Clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()    
    
    
    # Knowledge base status
    if st.session_state.knowledge_base_loaded:
        pass
        
    
    # Vectorstore status (for GPT)
    if "GPT" in model_option:
        st.session_state.vectorstore
    
    
    # API key status
    st.sidebar.subheader("🔑 API Key Status")
    if "GPT" in model_option:
        if os.getenv("OPENAI_API_KEY"):
            st.sidebar.success("✅ OpenAI API Key loaded")
        else:
            st.sidebar.error("❌ OpenAI API Key not found in .env")
    else:
        if os.getenv("GEMINI_API_KEY"):
            st.sidebar.success("✅ Gemini API Key loaded")
        else:
            st.sidebar.error("❌ Gemini API Key not found in .env")

def display_chat_interface():
    """Display the ChatGPT-like chat interface"""
    st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0;
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin: 10px 0;
    }
    
    .message-content {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 18px;
        font-size: 14px;
        line-height: 1.4;
    }
    
    .user-content {
        background-color: #007bff;
        color: white;
        margin-left: 10px;
    }
    
    .bot-content {
        background-color: #e9ecef;
        color: #333;
        margin-right: 10px;
    }
    
    .message-icon {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    .user-icon {
        background-color: #007bff;
        color: white;
    }
    
    .bot-icon {
        background-color: #28a745;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-content user-content">{message["content"]}</div>
                    <div class="message-icon user-icon">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div class="message-icon bot-icon">🤖</div>
                    <div class="message-content bot-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Paystack Customer Support Bot",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 Paystack Customer Support Bot")
    st.markdown("*Nigeria's trusted digital payments platform support*")
    st.markdown("---")
    
    # Setup sidebar
    setup_sidebar()
    
    # Main chat interface
    st.subheader("Paystack Customer Support Bot")
    
    # Display chat messages
    display_chat_interface()
    
    # Chat input form to avoid the session state error
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message here...",
            placeholder="Ask me about Paystack payments, fees, KYC, API integration, or any other queries...",
            key="user_message_input"
        )
        
        # Send button
        send_button = st.form_submit_button("Send", type="primary")
        
        # Process user input
        if send_button and user_input.strip():
            # Check API key availability
            if "GPT" in st.session_state.selected_model:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("OpenAI API key not found in environment variables. Please add OPENAI_API_KEY to your .env file.")
                    return
            else:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("Gemini API key not found in environment variables. Please add GEMINI_API_KEY to your .env file.")
                    return
            
            # Check if knowledge base is loaded
            if not st.session_state.knowledge_base_loaded:
                st.error("Knowledge base could not be loaded. Please ensure 'Paystack_FAQ_and_Documentation.txt' is in the same directory as this script.")
                return
            
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate bot response
            with st.spinner("Thinking..."):
                bot = CustomerSupportBot()
                
                if "GPT" in st.session_state.selected_model:
                    # Use OpenAI with RAG
                    if st.session_state.vectorstore:
                        context = bot.retrieve_relevant_context(user_input, st.session_state.vectorstore)
                        response = bot.generate_response_openai(user_input, context, api_key)
                    else:
                        # Fall back to using raw knowledge content
                        response = bot.generate_response_openai(user_input, st.session_state.knowledge_content, api_key)
                else:
                    # Use Gemini with simple context
                    response = bot.generate_response_gemini(user_input, st.session_state.knowledge_content, api_key)
                
                # Add bot response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Rerun to display new messages
            st.rerun()
    
    # Information panel
    st.sidebar.subheader("🏦 Paystack Features")
    st.sidebar.write("• Payment processing")
    st.sidebar.write("• KYC verification")
    st.sidebar.write("• API integration")
    st.sidebar.write("• Transaction monitoring")
    st.sidebar.write("• Compliance support")
    
    st.sidebar.subheader("📞 Support Channels")
    st.sidebar.write("📧 support@paystack.com")
    st.sidebar.write("📞 +234 1 700 5000")
    st.sidebar.write("💬 Live Chat: 8AM-10PM WAT")

if __name__ == "__main__":
    main()