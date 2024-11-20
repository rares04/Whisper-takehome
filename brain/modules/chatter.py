import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import KNNFewShot
from typing import Optional

from models import ChatHistory, ChatMetadata
from .responder import ResponderModule
from .sentiment import SentimentModule
from .filter import FilterModule

class ChatterModule(dspy.Module):
    def __init__(self, examples: Optional[dict]):
        super().__init__()
        self.sentiment = SentimentModule()
        self.responder = ResponderModule()
        self.filter = FilterModule()
        self.examples = examples

    def optimize(self):
        """Uses a KNNFewShot optimizer to improve the responder module"""
        if self.examples is not None:
            dl = DataLoader()
            splits = dl.train_test_split(self.examples, train_size=0.2)
            trainset = splits['train']
            devset = splits['test']

            knn_optimizer = KNNFewShot(k=3, trainset=trainset)

            self.responder = knn_optimizer.compile(student=self.responder, trainset=trainset, valset=devset)

    def forward(
        self,
        chat_metadata: ChatMetadata,
        chat_history: ChatHistory,
    ):
        # Determine sentiment
        sentiment = self.sentiment(chat_history=chat_history).sentiment

        # Generate response
        message = self.responder(chat_metadata=chat_metadata, chat_history=chat_history, sentiment=sentiment).output

        # Filter response
        return self.filter(message=message)