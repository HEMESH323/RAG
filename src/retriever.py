from src.utils import logger

class DocumentRetriever:
    """Handles retrieval of relevant documents from the vector store."""
    
    def __init__(self, vector_store, k: int = 6):
        self.vector_store = vector_store
        self.k = k

    def get_relevant_documents(self, query: str):
        """Retrieves top k relevant documents for a given query."""
        logger.info(f"Retrieving top {self.k} documents for query: {query}")
        return self.vector_store.similarity_search(query, k=self.k)
    
    def as_retriever(self):
        """Returns the vector store as a retriever object."""
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k}
        )

