import os
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError
from typing import Any
from client.response_classes import TextDelta, TokenUsage, StreamEvent, StreamEventType
from dotenv import load_dotenv
import asyncio

load_dotenv()


class LLMClient:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3

    def get_client(self):
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=os.environ.get("LLM_PROVIDER_BASE_URL"),
            )
        return self._client

    async def close(self):
        if self._client:
            self._client.close()
            self._client = None

    async def chat_completion(
        self, messages: list[dict[str, Any]], stream: bool = True
    ):
        client = self.get_client()
        kwargs = {
            "model": os.environ.get("DEFAULT_MODEL"),
            "messages": messages,
            "stream": stream,
        }
        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"Rate limit erro: {e}"
                    )
                    return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"API Connection error: {e}"
                    )
                    return
            except APIError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR, error=f"API error: {e}"
                )
                return

    async def _stream_response(self, client, kwargs):
        response = await client.chat.completions.create(**kwargs)
        finish_reason: str | None = None
        token_usage: TokenUsage | None = None

        async for response_chunk in response:
            if response_chunk.usage:
                token_usage = TokenUsage(
                    prompt_tokens=response_chunk.usage.prompt_tokens,
                    completion_tokens=response_chunk.usage.completion_tokens,
                    total_tokens=response_chunk.usage.total_tokens,
                    cached_tokens=response_chunk.usage.prompt_tokens_details.cached_tokens,
                )

            if not response_chunk.choices:
                continue

            choice = response_chunk.choices[0]
            choice_delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if choice_delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(choice_delta.content),
                )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            token_usage=token_usage,
        )

    async def _non_stream_response(self, client, kwargs):
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        token_usage = None
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            token_usage=token_usage,
        )
