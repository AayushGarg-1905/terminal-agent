import sys

from agent.agent import Agent
from agent.events import AgentEventType
from client.llm_client import LLMClient
import click
import asyncio
from typing import Any
from pathlib import Path
from config.config import Config
from config.loader import load_config
from ui.ui import UI, get_console

console = get_console()


class CLI:
    def __init__(self,config:Config):
        self.agent: Agent | None = None
        self.config = config
        self.ui = UI(config,console)

    async def run_single(self, message: str):
        async with Agent(self.config) as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self):
        self.ui.print_welcome()
        async with Agent(self.config) as agent:
            self.agent = agent
            while True:
                try:
                    user_input = console.input("\n[user]>[/user]").strip()
                    if not user_input:
                        continue
                    await self._process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim] Use /exit to quit[/dim]")
                except EOFError:
                    break
        console.print("\n[dim]Goodbye!![/dim]")

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
                tool = self.agent.session.tool_registry.get(tool_name)
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
                tool = self.agent.session.tool_registry.get(tool_name)
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
                    event.data.get("diff"),
                    event.data.get("truncated", False),
                )
        return final_response


@click.command()
@click.option("--prompt", required=False)
@click.option("--cwd","--c",type=click.Path(exists=True, file_okay=False, path_type=Path),help="Current working directory")
def main(prompt: str | None,cwd:Path | None):
    try:
        config = load_config(cwd=cwd)
    except Exception as e:
        console.print(f"Configuration error: {e}", style="error")
        sys.exit(1)
    config_errors = config.validate()
    if config_errors:
        for error in config_errors:
            console.print(f"Configuration error: {error}", style="error")
        sys.exit(1)
    # messages = [{"role":"user","content":prompt}]
    cli = CLI(config)
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


main()
