import sys

from agent.agent import Agent
from agent.events import AgentEventType
from client.llm_client import LLMClient
import click
import asyncio
from typing import Any

from ui.ui import UI, get_console

console = get_console()


class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.ui = UI(console)

    async def run_single(self, message: str):
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self, message: str):
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None
        async for event in self.agent.run(message):
            # print(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.ui.begin_assistant()
                    assistant_streaming = True
                self.ui.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_streaming:
                    self.ui.end_assistant()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "unknown error occured")
                self.ui.error_assistant(error)
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown tool")
                tool_kind = None
                tool = self.agent.tool_registry.get(tool_name)
                if tool:
                    tool_kind = tool.kind.value
                self.ui.tool_call_start(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown tool")
                tool_kind = None
                tool = self.agent.tool_registry.get(tool_name)
                if tool:
                    tool_kind = tool.kind.value
                self.ui.tool_call_complete(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("success", False),
                    event.data.get("output", ""),
                    event.data.get("error"),
                    event.data.get("metadata"),
                    event.data.get("truncated", False),
                )
        return final_response


@click.command()
@click.option("--prompt", required=False)
def main(prompt: str | None):
    cli = CLI()
    # messages = [{"role":"user","content":prompt}]
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)


main()
