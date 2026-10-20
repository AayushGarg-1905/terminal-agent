import abc
from enum import Enum
from typing import Any
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field
from pydantic.json_schema import model_json_schema
from pathlib import Path
import difflib

class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"


@dataclass
class FileDiff:
    path:Path
    old_content:str
    new_content:str
    is_new_file:bool = False
    is_deletion:bool = False

    def to_diff(self):
        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)

        if old_lines and not self.old_content[-1].endswith("\n"):
            old_lines[-1]+="\n"
        if new_lines and not self.new_content[-1].endswith("\n"):
            new_lines[-1]+="\n"

        old_name = "/dev/null" if self.is_new_file else str(self.path)
        new_name = "/dev/null" if self.is_deletion else str(self.path)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_name,
            tofile=new_name,
        )

        return "".join(diff)

@dataclass
class ToolInvocation:
    cwd: Path
    params: dict[str, Any]


@dataclass
class ToolConfirmation:
    tool_name: str
    params: dict[str, Any]
    description: str


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    diff: FileDiff | None = None
    
    @classmethod
    def error_result(cls, error: str, output: str = "",**kwargs):
        return cls(success=False, output=output, error=error,**kwargs)

    @classmethod
    def success_result(cls, output: str = "",**kwargs):
        return cls(success=True, output=output, **kwargs)

    def to_model_output(self):
        if self.success:
            return self.output
        return f"Error: {self.error} \n\n Output:\n{self.output}"

class Tool(abc.ABC):
    name: str = "base_tool"
    description: str = "Base Tool Description"
    kind: ToolKind = ToolKind.READ

    def __init__(self):
        pass

    @property
    def schema(self) -> dict[str, Any] | type["BaseModel"]:
        raise NotImplementedError("Tool must define a schema")

    @abc.abstractmethod
    def execute(self, invocation: ToolInvocation) -> ToolResult:
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    errors.append(f"Parameter '{field}': {msg}")

                return errors
            except Exception as e:
                return [str(e)]

        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY,
        }

    def get_confirmation(self, invocation: ToolInvocation):
        if not self.is_mutating(invocation.params):
            return None

        return ToolConfirmation(
            tool_name=self.name, params=invocation.params, description=self.description
        )

    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.schema

        if isinstance(schema, type) and issubclass(schema, BaseModel):

            json_schema = model_json_schema(schema, mode="serialization")

            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            }

        if isinstance(schema, dict):
            result = {
                "name": self.name,
                "description": self.description,
            }

            if "parameters" in schema:
                result["parameters"] = schema["parameters"]
            else:
                result["parameters"] = schema

            return result

        raise ValueError(f"Invalid schema type for tool {self.name}: {type(schema)}")
