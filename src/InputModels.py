from pydantic import BaseModel, RootModel
from typing import Dict, List


class ParameterType(BaseModel):
    """Model representing the type of a parameter."""
    type: str


class FunctionModel(BaseModel):
    """Model representing a function's metadata."""
    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType


class FunctionsLibrary(RootModel):
    """Model representing a library of functions."""
    root: List[FunctionModel]


class Prompt(BaseModel):
    """Model representing a prompt containing text."""
    prompt: str


class PromptLibrary(RootModel):
    """Model representing a library of prompts."""
    root: List[Prompt]
