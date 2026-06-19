from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("docs/python_notes.txt")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print("Total Chunks:", len(chunks))

for chunk in chunks:
    print("--------------------")
    print(chunk.page_content)