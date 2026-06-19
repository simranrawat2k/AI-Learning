from langchain_community.document_loaders import TextLoader

loader = TextLoader("docs/python_notes.txt")

documents = loader.load()

print(documents)