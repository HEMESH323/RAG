from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils import logger

class TextSplitter:
    """Splits documents into smaller text chunks."""
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 400):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of Documents into chunks."""
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks
