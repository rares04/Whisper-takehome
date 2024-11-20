import dspy

from models import ChatHistory

class SentimentSignature(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You need to classify the emotion of the FAN based on the chat history and provide the sentiment as output
    """

    chat_history: ChatHistory = dspy.InputField(
        prefix="Chat History: ",
        desc="the chat history; who sent the message, content, when the message was sent"
    )

    sentiment: str = dspy.OutputField(
        prefix="Sentiment:",
        desc="sentiment/vibe of the fan",
    )