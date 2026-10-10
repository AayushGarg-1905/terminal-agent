from dotenv import load_dotenv
from dataclasses import dataclass
import os
from prompts.system_prompt import get_system_prompt
from utils.tokenizer import count_tokens

load_dotenv()

@dataclass
class MessageItem:
    role:str
    content:str
    token_count:int | None = None

    def to_dict(self):
        result = {"role":self.role}

        if self.content:
            result["content"] = self.content
        return result


class ContextManager:
    def __init__(self):
        self._system_prompt = get_system_prompt()
        self._model_name = os.environ.get("DEFAULT_MODEL")
        self._messages: list[MessageItem] = []

    def add_user_message(self, content:str):
        user_message = MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(content)
        )
        self._messages.append(user_message)

    def add_assistant_message(self, content:str):
        assistant_message = MessageItem(
            role="assistant",
            content=content,
            token_count=count_tokens(content or "")
        )
        self._messages.append(assistant_message)

    def get_messages(self):
        messages = []

        if self._system_prompt:
            messages.append({
                "role":"system",
                "content":self._system_prompt
            })

        for item in self._messages:
            messages.append(item.to_dict())

        return messages