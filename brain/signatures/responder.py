import dspy

from models import ChatHistory, ChatMetadata

class Responder(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You are deciding on what your message should be.
    You know when the conversation started, how long the conversation has been going for and the sentiment of the fan.
    """

    chat_metadata: ChatMetadata = dspy.InputField(
        prefix="Chat Metadata: ",
        desc="current time, when the conversation has started, how long the conversation has been going for"
    )
    chat_history: ChatHistory = dspy.InputField(
        prefix="Chat History: ",
        desc="the chat history; who sent the message, content, when the message was sent"
    )
    sentiment: str = dspy.InputField(
        prefix="Sentiment: ",
        desc="sentiment/vibe of the fan"
    )

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="the unfiltered text of the message you will send to the fan.",
    )