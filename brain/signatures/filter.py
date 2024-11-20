import dspy

class FilterSignature(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You take into consideration topics that may not be suitable for discussion
    You keep responses free of mentions of social media platforms (except OnlyFans) and interactions suggesting in-person meetings with fans
    """
    
    message: str = dspy.InputField(
        prefix="Unfiltered message: ",
        desc="the unfiltered text of the message you will send to the fan."
    )

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )