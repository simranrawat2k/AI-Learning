from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load variables from .env
load_dotenv()

# Create AI model
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Ask a question
response = llm.invoke("What is Python?")

# Print answer
print(response.content)