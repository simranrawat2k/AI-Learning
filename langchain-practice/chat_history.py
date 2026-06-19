from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=300
)

# Conversation memory
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

print("Chatbot started. Type 'exit' to stop.\n")

while True:
    # 1. Take user input
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # 2. Add user message to memory
    messages.append(
        HumanMessage(content=user_input)
    )

    # 3. Get AI response
    response = llm.invoke(messages)

    # 4. Print response
    print("AI:", response.content)

    # 5. Save AI response to memory
    messages.append(
        AIMessage(content=response.content)
    )