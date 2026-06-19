from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Open Existing ChromaDB
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Convert Documents -> Text
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {question}
    """
)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# Ask Question
response = rag_chain.invoke(
    "What is a virtual environment?"
)

print(response.content)