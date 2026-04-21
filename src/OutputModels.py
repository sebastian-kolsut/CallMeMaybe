from pydantic import BaseModel, RootModel
from typing import Dict, List


class FunctionCall(BaseModel):
    """Model representing a generated function call."""
    prompt: str
    name: str
    parameters: Dict[str, float | str | bool]


class FunctionCallsLibrary(RootModel):
    """Model representing a library of function calls."""
    root: List[FunctionCall]
