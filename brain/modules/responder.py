import dspy

from signatures.responder import Responder
from models import ChatHistory, ChatMetadata

class ResponderModule(dspy.Module):
    def __init__(self):
        super().__init__()

        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We",
        )
        self.prog = dspy.TypedChainOfThought(Responder, reasoning=reasoning)
    
    def forward(
        self,
        chat_metadata: dict,
        chat_history: dict,
        sentiment: str
    ):
        return self.prog(
            chat_metadata=ChatMetadata.parse_obj(chat_metadata),
            chat_history=ChatHistory.parse_obj(chat_history),
            sentiment=sentiment
        )