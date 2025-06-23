from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.llms import Cohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

app = Flask(__name__)


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="med-index"
)
retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)

# LLM setup
llm = Cohere(
    model="command-r-plus",  
    temperature=0.0,
    max_tokens=300,
    cohere_api_key=COHERE_API_KEY
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a medical assistant AI. Using only the information in the context below:

    - Do NOT include any code, function definitions, or explanations.
    - Do NOT make up anything not stated in the context.
    - If the answer is not explicitly available, reply with:
    "I don't know based on the provided information."

    Context:
    {context}

    Question: {input}
    Only reply with the answer sentence, nothing else.
    """
)


def clean_doc(text):
    if "Please answer the next question" in text:
        text = text.split("Please answer the next question")[0]
    return text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please enter a valid question."})

    try:
        # Retrieve and clean documents
        docs = retriever.invoke(question)
        for i, doc in enumerate(docs):
            print(f"\n--- Retrieved Document {i+1} ---\n")
            print(doc.page_content)

        # Clean context
        cleaned_context = "\n\n".join([clean_doc(doc.page_content) for doc in docs])

        # Format prompt as messages for Cohere
        messages = prompt.format_messages(context=cleaned_context, input=question)

        # Optional: Debug the final prompt
        print("\n=== Final Prompt Sent to LLM ===")
        for msg in messages:
            print(f"{msg.type.upper()}: {msg.content}")

        # Invoke the model
        answer = llm.invoke(messages)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)