from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class ChatMetadata(BaseModel):
    """Holds metadata information such as start time of the conversation"""
    start_time: datetime = datetime.now()

    def __str__(self):
        now = datetime.now()
        formatted_current_time = now.strftime("%A, %m/%d/%Y")
        formatted_start_time = self.start_time.strftime("%A, %m/%d/%Y")

        time_difference = now - self.start_time

        formatted_start_time = self.start_time.strftime("%A, %m/%d/%Y")
        return f"Current date and time is {formatted_current_time}. The conversation started {formatted_start_time}, which means the conversation has been going for {time_difference.days + 1} day(s)"

    def model_dump_json(self, **kwargs):
        return str(self)

class ChatMessage(BaseModel):
    from_creator: bool
    content: str
    sent_at: Optional[datetime] = None

    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        message = f"{role}: {self.content}"
        if(self.sent_at is not None):
            message += f" (Sent at: {self.sent_at})"
        return message

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []

    def __str__(self):
        messages = []
        for i, message in enumerate(self.messages):
            message_str = str(message)
            # if i == len(self.messages) - 1 and not message.from_creator:
            #     message_str = (
            #         "(The fan just sent the following message which your message must respond to): "
            #         + message_str
            #     )
            messages.append(message_str)
        return "\n".join(messages)
    
    def model_dump_json(self, **kwargs):
        return str(self)
    