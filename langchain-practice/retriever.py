from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Open existing database
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Convert to retriever
retriever = vector_store.as_retriever()

# Ask question
results = retriever.invoke(
    "What is a virtual environment?"
)

# Print results
for doc in results:
    print("----------------")
    print(doc.page_content)