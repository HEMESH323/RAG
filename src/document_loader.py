import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.utils import logger

class PDFLoader:
    """Handles loading of PDF documents."""
    
    @staticmethod
    def load_pdfs(file_paths: List[str]) -> List[Document]:
        """Loads multiple PDF files and returns a list of LangChain Documents."""
        all_documents = []
        for file_path in file_paths:
            try:
                logger.info(f"Loading document: {file_path}")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_documents)} pages from {len(file_paths)} documents.")
        return all_documents
