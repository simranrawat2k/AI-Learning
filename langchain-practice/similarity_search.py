from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

results = vector_store.similarity_search(
    "What is a virtual environment?",
    k=2
)

for result in results:
    print("----------------")
    print(result.page_content)