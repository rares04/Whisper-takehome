import json
import dspy
import os
from models import ChatHistory, ChatMessage, ChatMetadata



def load_examples():
    """Load examples form conversations.json and converts them into dspy.Example objects"""
    cwd = os.getcwd()
    conversations_path = os.path.join(cwd, "training_data", "conversations.json")
    with open(conversations_path, "r") as file:
        dataset = json.load(file)

    examples = []
    for entry in dataset:
        chat_metadata = ChatMetadata()
        chat_history = ChatHistory()
        for message in entry['chat_history']['messages']:
            chat_history.messages.append(
                ChatMessage(
                    from_creator=message['from_creator'],
                    content=message['content']
                )
            )
        examples.append(
            dspy.Example(
                chat_metadata=chat_metadata,
                chat_history=chat_history,
                sentiment="N/A",
                output=f"Your message: {entry["output"]}"
            ).with_inputs("chat_metadata", "chat_history", "sentiment")
        )

    return examples
