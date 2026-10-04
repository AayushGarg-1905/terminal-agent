import os
from openai import AsyncOpenAI
from typing import Any
from client.response_classes import TextDelta, TokenUsage, StreamEvent, StreamEventType
from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def get_client(self):
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=os.environ.get("LLM_PROVIDER_BASE_URL")
            )
        return self._client

    async def close(self):
        if self._client:
            self._client.close()
            self._client = None

    async def chat_completion(self, messages:list[dict[str,Any]], stream:bool=True):
        client = self.get_client()
        kwargs = {
            "model": os.environ.get("DEFAULT_MODEL"),
            "messages":messages,
            "stream":stream
        }
        if stream:
            pass
        else:
            event = await self._non_stream_response(client,kwargs)
            yield event
        return

    async def _non_stream_response(self, client, kwargs):
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content = message.content)

        token_usage = None
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens= response.usage.prompt_tokens,
                completion_tokens= response.usage.completion_tokens,
                total_tokens= response.usage.total_tokens,
                cached_tokens= response.usage.prompt_tokens_details.cached_tokens
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta= text_delta,
            finish_reason= choice.finish_reason,
            token_usage=token_usage
        )

        
