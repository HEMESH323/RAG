import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
from src.document_loader import PDFLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from src.chatbot import ChatbotManager
from src.memory import MemoryManager
from src.utils import logger

# Load environment variables
load_dotenv(override=True)

# Page configuration
st.set_page_config(page_title="AI Multi-Document Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* New Chat Button Styling */
    div.stButton > button:first-child {
        border-radius: 20px;
        height: 3em;
        width: 100%;
    }
    /* Chat Item Styling */
    .chat-item-current {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    /* Sidebar Headers */
    .sidebar-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initializes streamlit session state variables."""
    if "chats" not in st.session_state:
        # Dictionary to store chats: {id: {"name": str, "messages": [], "memory": MemoryManager}}
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def create_new_chat():
    """Creates a new chat session."""
    import uuid
    chat_id = str(uuid.uuid4())
    memory_manager = MemoryManager()
    st.session_state.chats[chat_id] = {
        "name": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "memory": memory_manager.get_memory()
    }
    st.session_state.current_chat_id = chat_id
    # Re-initialize chatbot for the new session if vector store exists
    if st.session_state.vector_store:
        retriever = DocumentRetriever(st.session_state.vector_store).as_retriever()
        st.session_state.chatbot = ChatbotManager(retriever, memory_manager.get_memory())
    else:
        st.session_state.chatbot = None

def process_documents(uploaded_files):
    """Processes uploaded PDF documents and initializes the chatbot."""
    with st.spinner("Processing documents... This might take a while."):
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # 1. Load Documents
        loader = PDFLoader()
        documents = loader.load_pdfs(file_paths)
        
        # 2. Split Text
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            st.error("No text could be extracted from the uploaded documents. Please ensure they contain readable text.")
            return

        # 3. Initialize Embeddings
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()
        
        # 4. Create Vector Store
        vector_store_manager = VectorStoreManager(embeddings)
        st.session_state.vector_store = vector_store_manager.create_vector_store(chunks)
        
        if not st.session_state.vector_store:
            st.error("Failed to create the vector store. Please try again.")
            return

        # 5. Initialize Memory & Chatbot for first chat if none exists
        if not st.session_state.chats:
            create_new_chat()
        else:
            # Update current chatbot with new retriever
            current_chat = st.session_state.chats[st.session_state.current_chat_id]
            retriever = DocumentRetriever(st.session_state.vector_store).as_retriever()
            st.session_state.chatbot = ChatbotManager(retriever, current_chat["memory"])
            
        st.session_state.processed_files = [f.name for f in uploaded_files]
        st.success("Documents processed successfully! You can now start chatting.")

def main():
    initialize_session_state()
    
    # Sidebar Redesign
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><span>🤖</span> RAG Assistant</div>', unsafe_allow_html=True)
        
        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()
            
        st.markdown('<div style="display: flex; align-items: center; gap: 8px; font-weight: bold; margin-top: 20px;"><span style="font-size: 20px;"></span>My Coversations</div>', unsafe_allow_html=True)
        
        if st.session_state.chats:
            for chat_id, chat_info in st.session_state.chats.items():
                is_current = chat_id == st.session_state.current_chat_id
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    button_label = f"{'⭐' if is_current else '💬'} {chat_info['name']}"
                    # Custom button styling for current chat items could be complex in pure Streamlit,
                    # so we use standard primary/secondary or simple markdown boxes if needed.
                    if st.button(button_label, key=f"select_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        if st.session_state.vector_store:
                            retriever = DocumentRetriever(st.session_state.vector_store).as_retriever()
                            st.session_state.chatbot = ChatbotManager(retriever, chat_info["memory"])
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}"):
                        del st.session_state.chats[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.current_chat_id = next(iter(st.session_state.chats)) if st.session_state.chats else None
                            st.session_state.chatbot = None
                        st.rerun()

        st.markdown("---")
        st.markdown('<div style="display: flex; align-items: center; gap: 8px; font-weight: bold; margin-bottom: 10px;"><span style="font-size: 20px;">📄</span> Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
        
        if st.button("Process Documents") and uploaded_files:
            process_documents(uploaded_files)
            
        if st.session_state.processed_files:
            st.write("### Uploaded Files:")
            for file_name in st.session_state.processed_files:
                st.write(f"- {file_name}")
                
        if st.button("Clear History of This Chat"):
            if st.session_state.current_chat_id:
                chat = st.session_state.chats[st.session_state.current_chat_id]
                chat["memory"].clear()
                chat["messages"] = []
                st.rerun()

        st.divider()
        st.header("🎙️ Voice Input")
        voice_prompt = speech_to_text(
            language='en', 
            start_prompt="Click to Speak 🎙️",
            stop_prompt="Stop Recording 🛑",
            just_once=True, 
            key='STT'
        )

    # Chat Interface
    chat_container = st.container()
    
    if st.session_state.current_chat_id:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        with chat_container:
            for message in current_chat["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        with st.expander("Show Sources"):
                            for source in message["sources"]:
                                st.write(f"- {source}")
    else:
        with chat_container:
            st.info("👋 Welcome! Upload documents and click 'Process' or start a 'New Chat' to begin.")

    chat_prompt = st.chat_input("Ask a question about your documents...")
    
    # Check if either chat input or voice input is provided
    prompt = chat_prompt or voice_prompt

    if prompt:
        if not st.session_state.chatbot:
            st.error("Please upload and process documents first!")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            current_chat = st.session_state.chats[st.session_state.current_chat_id]
            current_chat["messages"].append({"role": "user", "content": prompt})
            
            # Auto-rename chat based on first message
            if current_chat["name"].startswith("Chat ") and len(current_chat["messages"]) == 1:
                current_chat["name"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

            # Generate AI response
            with st.chat_message("assistant"):
                # Use a placeholder for the streaming response
                response_placeholder = st.empty()
                full_response = ""
                source_docs = []
                
                # The chatbot now returns a generator
                response_stream = st.session_state.chatbot.ask(prompt)
                
                # Stream the response
                for chunk in response_stream:
                    if isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    else:
                        # This is the list of source documents yielded at the end
                        source_docs = chunk
                
                # Final display without the cursor
                response_placeholder.markdown(full_response)
                
                sources = []
                for doc in source_docs:
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page_num = doc.metadata.get("page", "Unknown")
                    if page_num != "Unknown":
                        sources.append(f"{source_name} (Page {page_num + 1})")
                    else:
                        sources.append(source_name)
                
                sources = list(set(sources))
                if sources:
                    with st.expander("Show Sources"):
                        for source in sources:
                            st.write(f"- {source}")
                
                current_chat["messages"].append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })

if __name__ == "__main__":
    main()
