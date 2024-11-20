import dspy

from brain.signatures.filter import FilterSignature

class FilterModule(dspy.Module):
    """Ensures the message avoids specific topics that may not be suitable"""
    def __init__(self):
        super().__init__()
        filter_reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide wheter the message needs filtering or not. We",
        )
        self.filter = dspy.TypedChainOfThought(FilterSignature, reasoning=filter_reasoning)
    
    def forward(
        self,
        message: str
    ):
        return self.filter(message=message)
