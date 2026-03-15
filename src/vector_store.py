import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils import logger

class VectorStoreManager:
    """Manages the FAISS vector store."""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None

    def create_vector_store(self, chunks: List[Document], save_path: str = "faiss_index"):
        """Creates a FAISS vector store from text chunks and saves it locally."""
        if not chunks:
            logger.warning("No document chunks provided. Skipping vector store creation.")
            return None
            
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(save_path)
        logger.info(f"Vector store saved to {save_path}.")
        return self.vector_store

    def load_vector_store(self, load_path: str = "faiss_index"):
        """Loads a FAISS vector store from local storage."""
        if os.path.exists(load_path):
            logger.info(f"Loading vector store from {load_path}...")
            self.vector_store = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
            return self.vector_store
        else:
            logger.warning(f"No vector store found at {load_path}.")
            return None
