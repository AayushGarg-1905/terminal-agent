import json

from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text
from typing import Any
from rich.panel import Panel
from rich.table import Table

AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Code / blocks
        "code": "white",
    }
)


_console: Console | None = None


def get_console():
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
    return _console


class UI:
    def __init__(self, console: Console | None = None):
        self.console = console or get_console()
        self._assistant_stream_open = False
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}

    def begin_assistant(self):
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant")))
        self._assistant_stream_open = True

    def end_assistant(self) -> None:
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False

    def error_assistant(self, error: str):
        self.console.print(error, style="error")

    def stream_assistant_delta(self, content: str):
        self.console.print(content, end="", markup=False)

    def tool_call_start(
        self,
        call_id: str,
        name: str,
        tool_kind: str | None,
        arguments: dict[str, Any],
    ) -> None:
        self._tool_args_by_call_id[call_id] = arguments

        kind = getattr(tool_kind, "value", tool_kind)
        kind = str(kind).lower() if kind else ""

        valid_kinds = {"read", "write", "shell", "network", "memory", "mcp"}
        border_style = f"tool.{kind}" if kind in valid_kinds else "tool"

        details = Table.grid(padding=(0, 1))
        details.add_column(style="muted", no_wrap=True)
        details.add_column()

        if arguments:
            for key, value in arguments.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                else:
                    value = str(value)

                if len(value) > 120:
                    value = value[:117] + "..."

                details.add_row(
                    Text(str(key)),
                    Text(value, style="code", overflow="fold"),
                )
        else:
            details.add_row(Text("Arguments"), Text("None", style="muted"))

        title = Text()
        title.append(" TOOL ", style="bold white on grey23")
        title.append(f" {name} ", style="bold")

        subtitle = Text("RUNNING", style="bold yellow")

        details.add_row(
            Text("Call ID"),
            Text(call_id, style="dim"),
        )

        panel = Panel(
            details,
            title=title,
            subtitle=subtitle,
            border_style=border_style,
            padding=(0, 1),
            expand=False,
        )

        self.console.print(panel)

    
    def tool_call_complete(
        self,
        call_id: str,
        name: str,
        tool_kind: str | None,
        success: bool,
        output: str,
        error: str | None,
        metadata: dict[str, Any] | None,
        truncated: bool,
    ) -> None:
        kind = getattr(tool_kind, "value", tool_kind)
        kind = str(kind).lower() if kind else ""

        valid_kinds = {"read", "write", "shell", "network", "memory", "mcp"}
        border_style = f"tool.{kind}" if kind in valid_kinds else "tool"

        status = "SUCCESS" if success else "FAILED"
        status_style = "bold green" if success else "bold bright_red"

        # Build the panel heading.
        title = Text()
        title.append(" TOOL ", style="bold white on grey23")
        title.append(f" {name} ", style="bold")

        subtitle = Text(status, style=status_style)

        # Build the details table.
        details = Table.grid(padding=(0, 1))
        details.add_column(style="muted", no_wrap=True)
        details.add_column()

        details.add_row(
            Text("Call ID"),
            Text(call_id, style="dim"),
        )

        if metadata:
            details.add_row(
                Text("Metadata"),
                Text(
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    style="dim",
                    overflow="fold",
                ),
            )

        # Prefer the explicit error field for failed calls.
        result_text = error if not success and error else output
        result_text = result_text or "(No output)"

        # Prevent huge tool outputs from overwhelming the terminal.
        max_output_chars = 5000
        output_truncated = len(result_text) > max_output_chars

        if output_truncated:
            result_text = (
                result_text[:max_output_chars]
                + "\n... [display truncated]"
            )

        if truncated:
            result_text += "\n... [tool output was truncated]"

        details.add_row(
            Text("Output" if success else "Error"),
            Text(
                result_text,
                style="code" if success else "error",
                overflow="fold",
            ),
        )

        panel = Panel(
            details,
            title=title,
            subtitle=subtitle,
            border_style=border_style if success else "error",
            padding=(0, 1),
            expand=False,
        )

        self.console.print(panel)

        # Remove the stored arguments now that this call has completed.
        self._tool_args_by_call_id.pop(call_id, None)

