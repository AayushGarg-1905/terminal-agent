from agent.events import AgentEvent, AgentEventType
from agent.session import Session
from client.response_classes import StreamEventType, ToolCall, ToolResultMessage
from config.config import Config
import json

class Agent:
    def __init__(self,config:Config):
        self.config = config
        self.session = Session(config)

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        self.session.context_manager.add_user_message(message)

        final_response: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self):
        max_turns = self.config.max_turns
        # messages = [{"role":"user","content":"write a short para on cricket"}]
        for turn_num in range(max_turns):
            self.session.increment_turn()
            
            response_text = ""
            tool_schemas = self.session.tool_registry.get_schemas()
            tool_calls: list[ToolCall] = []
            tool_call_results: list[ToolResultMessage] = []

            async for event in self.session.client.chat_completion(
                self.session.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                stream=True,
            ):
                # print(event)
                if event.type == StreamEventType.TEXT_DELTA:
                    if event.text_delta:
                        content = event.text_delta.content
                        response_text += content
                        yield AgentEvent.text_delta(content)
                elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                    if event.tool_call:
                        tool_calls.append(event.tool_call)
                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(event.error or "Unknown error occured")
            formatted_tool_calls = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ]
            self.session.context_manager.add_assistant_message(
                response_text or "", formatted_tool_calls or None
            )

            if response_text:
                yield AgentEvent.text_complete(response_text)

            if not tool_calls:
                return
            
            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(
                    tool_call.call_id, tool_call.name, tool_call.arguments
                )
                result = await self.session.tool_registry.invoke(
                    tool_call.name, tool_call.arguments, self.config.cwd
                )

                yield AgentEvent.tool_call_complete(
                    tool_call.call_id, tool_call.name, result
                )

                tool_call_results.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.call_id,
                        content=result.to_model_output(),
                        is_error=not result.success,
                    )
                )

            for tool_result in tool_call_results:
                self.session.context_manager.add_tool_message(
                    tool_result.tool_call_id, tool_result.content
                )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and self.session.client:
            await self.session.client.close()
            self.session = None
