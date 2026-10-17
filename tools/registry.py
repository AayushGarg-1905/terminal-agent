from typing import Any
from pathlib import Path
from tools.base import Tool, ToolInvocation, ToolResult
import logging

from tools.builtin import get_all_builtin_tools
from tools.builtin.read_file import ReadFileTool

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self):
        self._tools:dict[str,Tool] = {}

    def register(self, tool:Tool):
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> bool:
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = []

        for tool in self._tools.values():
            tools.append(tool)

        return tools
    
    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self.get_tools()]

    def get(self, name:str):
        if name in self._tools:
            return self._tools[name]
        return None

    async def invoke(self, name:str, params:dict[str,Any],cwd:Path):
        tool = self.get(name)
        if tool is None:
            return ToolResult.error_result(
                f"Unknow tool: {name}",
                metadata={"tool_name":name}
            )

        validation_errors = tool.validate_params(params)

        if validation_errors:
            return ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={"tool_name":name, "validation_errors":validation_errors}
            )

        tool_invocation = ToolInvocation(
            params=params,
            cwd=cwd
        )

        try:

            result = await tool.execute(invocation=tool_invocation)
            return result
        except Exception as e:
            logger.exception(f"Tool {name} raise error {str(e)}")
            return ToolResult.error_result(
                f"Internal error: {str(e)}",
                metadata={"tool_name":name}
            )


def create_default_registry():
    tool_registry = ToolRegistry()
    for tool_class in get_all_builtin_tools():
        tool_registry.register(tool_class())
    return tool_registry

    