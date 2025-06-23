# Med_Buddy

A Retrieval-Augmented Generation (RAG) based medical assistant chatbot built with Flask, LangChain, Cohere LLM, Pinecone, and HuggingFace embeddings. It intelligently answers health-related queries by retrieving the most relevant context from a pre-built medical knowledge base, ensuring responses are accurate, safe, and grounded in the provided data.

📌 Features

    🔍 Context-Aware QA: Retrieves medical documents relevant to the user's query before answering.
    🤖 LLM Powered Responses: Uses Cohere's command-r-plus model to generate concise and safe answers.
    🧠 Semantic Search: Implements HuggingFace’s all-MiniLM-L6-v2 model for document embeddings.
    📦 Vector DB with Pinecone: Stores and retrieves document vectors using Pinecone’s managed vector index.
    🔐 Environment Variable Integration: Uses python-dotenv for secure key management.
    🌐 Flask Web Interface: Simple web frontend for asking medical queries.
    ⚠️ Safety Guardrails: LLM prompt restricts hallucinations and irrelevant answers.


🎯 Goal
This project aims to:

    Showcase a production-grade RAG pipeline for domain-specific chatbots.
    Enable safe and traceable AI use in sensitive domains like healthcare.
    Help students and developers understand how to combine retrieval + generation effectively.


🧰 Tech Stack

Category	Tech
Backend	Flask
Retrieval Engine	Pinecone
Embeddings	HuggingFace Transformers (all-MiniLM-L6-v2)
LLM	Cohere (command-r-plus)
RAG Framework	LangChain
Environment Vars	python-dotenv


🚀 Setup Instructions

1. Clone the Repository

2. Install Dependencies:
(pip install -r requirements.txt)

3. Set Up Environment Variables by creating a .env file in which set up:

PINECONE_API_KEY=your_pinecone_key and 
COHERE_API_KEY=your_cohere_key

5. Run the Flask App
python app.py


📝 Customization
    
    🔧 To update the knowledge base, ingest your own documents and re-index them using Pinecone.
    ✏️ Modify the prompt under ChatPromptTemplate to change the assistant’s behavior.
    🛡️ Add further validation layers for medical safety (e.g., model filters, approval systems).


⚠️ Disclaimer

This chatbot is for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

⭐️ Credits
    
    Cohere
    Pinecone
    LangChain
    HuggingFace
