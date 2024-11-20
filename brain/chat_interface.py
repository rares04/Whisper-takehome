from models import ChatMessage, ChatHistory, ChatMetadata
import dspy
from lms.together import Together
from examples_loader import load_examples

from datetime import datetime

from modules.chatter import ChatterModule

lm = Together(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1.2,
    stop=["<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"],
    # stop=["\n", "\n\n"],
)

dspy.settings.configure(lm=lm)

chat_history = ChatHistory()
chat_metadata = ChatMetadata()
chatter = ChatterModule(examples=load_examples())
chatter.optimize()  # KNNFewShot optimizer for the Responder module
while True:
    # Get user input
    user_input = input("You: ")

    # Append user input to chat history
    chat_history.messages.append(
        ChatMessage(
            from_creator=False,
            content=user_input,
            sent_at=datetime.now()
        ),
    )

    # Send request to endpoint
    response = chatter(chat_metadata=chat_metadata, chat_history=chat_history).output

    # Append response to chat history
    chat_history.messages.append(
        ChatMessage(
            from_creator=True,
            content=response,
            sent_at=datetime.now()
        ),
    )
    # Print response
    print()
    print("Response:", response)
    print()
    # uncomment this line to see the 
    # lm.inspect_history(n=5)