from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from src.utils import logger, get_env_variable

class ChatbotManager:
    """Manages the Chatbot logic using Google Gemini RAG."""
    
    def __init__(self, retriever, memory):
        api_key = get_env_variable("GOOGLE_API_KEY")
        logger.info("Initializing ChatbotManager with Gemini...")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=api_key,
            temperature=0.3,
            streaming=True,
            max_retries=6  # Increased retries for handling transient 429 errors (RPM)
        )
        
        self.memory = memory
        self.retriever = retriever
        
        # System prompt template
        self.template = """
        You are an intelligent assistant. Answer strictly using the provided context. 
        If the answer is not found in the context, say you don't know and do not try to make up an answer.
        
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        
        Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )

    def ask(self, query: str):
        """Asks a question and yields the answer with source documents."""
        logger.info(f"Asking chatbot: {query}")
        
        try:
            result = self.chain.invoke({"question": query})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Simulate streaming for the UI to handle smoothly
            import time
            for word in answer.split(" "):
                yield word + " "
                time.sleep(0.02)
                
            yield source_docs
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error calling Gemini API: {error_str}")
            
            if "429" in error_str:
                if "quota" in error_str.lower() or "limit" in error_str.lower():
                    error_msg = ("⚠️ **Gemini Quota Exceeded.**\n\n"
                                 "You have reached the daily limit for the Gemini Free Tier. "
                                 "Please update the `GOOGLE_API_KEY` in your `.env` file with a new key "
                                 "or wait for the quota to reset.")
                else:
                    error_msg = ("⚠️ **Gemini Rate Limit Hit.**\n\n"
                                 "Too many requests in a short time. Please wait a moment and try again.")
            else:
                error_msg = f"⚠️ **An unexpected error occurred:** {error_str}"
            
            yield error_msg
            yield []  # Empty source docs

