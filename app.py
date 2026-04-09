import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
        color: black;
    }
    .ai-message {
        background-color: #f5f5f5;
        border-left: 4px solid #2e7d32;
        color: black;
    }
    .document-retrieval {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f57c00;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# Header
st.markdown("# 📚 RAG Document Assistant")
st.markdown("Transform your documents into an intelligent knowledge base with AI-powered search and Q&A")

# Sidebar for PDF upload
with st.sidebar:
    st.markdown("## 📄 Document Management")
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf",
        help="Select a PDF file to create a knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("Processing PDF... This may take a moment..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    
                    # Split documents
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)
                    
                    # Create embeddings and vector store
                    embedding_model = MistralAIEmbeddings()
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                    )
                    
                    st.session_state.vectorstore = vectorstore
                    st.session_state.document_loaded = True
                    st.session_state.chat_history = []
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Document processed successfully!")
                    st.info(f"📊 Processed {len(chunks)} text chunks from {len(docs)} pages")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
    
    st.divider()
    
    # Display document status
    if st.session_state.document_loaded:
        st.success("✅ Document loaded and ready")
        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.document_loaded = False
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("⏳ Waiting for document upload...")
    
    st.divider()
    st.markdown("### ℹ️ How to use:")
    st.markdown("""
    1. **Upload** a PDF document
    2. **Process** the document
    3. **Ask questions** about the content
    4. Get AI-powered answers with source references
    """)

# Main chat interface
if st.session_state.document_loaded:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 💬 Ask Your Questions")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div style="width: 100%; color: black;">
                            <strong>👤 You</strong><br>
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message ai-message">
                        <div style="width: 100%; color: black;">
                            <strong>🤖 AI Assistant</strong><br>
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if "documents" in message:
                    with st.expander("📖 Source Documents"):
                        for i, doc in enumerate(message["documents"], 1):
                            st.markdown(f"**Document {i}:**")
                            st.markdown(f"_{doc}_")
                            st.divider()
    
    # Input section
    st.divider()
    
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Your question:",
            placeholder="Ask a question about your document...",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("Send", use_container_width=True, key="send_button")
    
    if send_button and user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        try:
            # Retrieve relevant documents
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
            )
            
            docs = retriever.invoke(user_query)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """You are a helpful AI assistant that answers questions based on the provided context.
                     If the answer is not in the context, say: "I could not find the answer in the document."
                     Provide clear and concise answers."""),
                    ("human", """Context: {context}
                     
Question: {question}""")
                ]
            )
            
            # Generate response
            llm = ChatMistralAI(model="mistral-small-2506")
            final_prompt = prompt.invoke({"context": context, "question": user_query})
            response = llm.invoke(final_prompt)
            
            # Add AI message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.content,
                "documents": [doc.page_content[:200] + "..." for doc in docs]
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

else:
    # Welcome screen
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>📖 Welcome to RAG Document Assistant</h2>
            <p style="font-size: 1.1rem; color: #666;">
                Upload a PDF document from the sidebar to get started.
            </p>
            <p style="font-size: 0.95rem; color: #999;">
                Once uploaded, you can ask questions and get intelligent answers
                based on the document content.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        ### 🎯 Features:
        - 📄 Upload and process PDF documents
        - 🔍 Semantic search across your documents
        - 🤖 AI-powered question answering
        - 📊 View retrieved source documents
        - 💾 Persistent chat history
        """)
