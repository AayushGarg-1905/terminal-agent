import os
from openai import AsyncOpenAI
from typing import Any
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
            await self._non_stream_chat_completion(client,kwargs)

    async def _non_stream_chat_completion(self, client, kwargs):
        response = await client.chat.completions.create(**kwargs)
        print(response.choices[0].message.content)
        
