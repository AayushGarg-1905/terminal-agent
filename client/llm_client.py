import os
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError
from typing import Any
from client.response_classes import (
    TextDelta,
    TokenUsage,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)
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
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str, Any]]):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                },
            }
            for tool in tools
        ]

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ):
        client = self.get_client()
        kwargs = {
            "model": os.environ.get("DEFAULT_MODEL"),
            "messages": messages,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

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
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"Rate limit erro: {e}"
                    )
                    return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"API Connection error: {e}"
                    )
                    return
            except APIError as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=f"API error: {e}")
                return

    async def _stream_response(self, client, kwargs):
        response = await client.chat.completions.create(**kwargs)
        finish_reason: str | None = None
        token_usage: TokenUsage | None = None
        tool_calls: dict[int, dict[str, Any]] = {}

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
            # print("#####")
            # print(choice_delta)
            # print("######")
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if choice_delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(choice_delta.content),
                )

            if choice_delta.tool_calls:
                for tool_call_delta in choice_delta.tool_calls:
                    idx = tool_call_delta.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_calls[idx]["name"] = tool_call_delta.function.name
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_calls[idx]["name"],
                                ),
                            )
                    if tool_call_delta.function.arguments:
                        tool_calls[idx][
                            "arguments"
                        ] += tool_call_delta.function.arguments
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            tool_call_delta=ToolCallDelta(
                                call_id=tool_calls[idx]["id"],
                                name=tool_calls[idx]["name"],
                                arguments_delta=tool_calls[idx]["arguments"],
                            ),
                        )

        for idx, tool_call in tool_calls.items():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=tool_call["id"],
                    name=tool_call["name"],
                    arguments=parse_tool_call_arguments(tool_call["arguments"]),
                ),
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
        tool_calls: list[ToolCall] = []
        if message.content:
            text_delta = TextDelta(content=message.content)

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=parse_tool_call_arguments(tc.function.arguments),
                    )
                )

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
