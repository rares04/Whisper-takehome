import dspy

from brain.signatures.sentiment import SentimentSignature
from models import ChatHistory

class SentimentModule(dspy.Module):
    """Determines the sentiment/vibe of the conversation"""
    def __init__(self):
        super().__init__()
        self.sentiment = dspy.TypedChainOfThought(SentimentSignature)
    
    def forward(
        self,
        chat_history: dict
    ):
        return self.sentiment(chat_history=ChatHistory.parse_obj(chat_history))