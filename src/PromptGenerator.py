from typing import Dict, Tuple, Any
from .InputModels import (
    FunctionsLibrary, ParameterType, PromptLibrary, FunctionModel
    )
from pydantic import ValidationError
import sys


class PromptGenerator:
    """
    Responsible for constructing and managing prompts for the LLM based on
    function definitions.
    """
    def __init__(self, file_name: str) -> None:
        """
        Initialize the PromptGenerator with function definitions from a JSON
        file.

        Args:
            file_name (str): Path to the JSON file containing function
                definitions.
        """
        try:
            with open(file_name, "r") as file:
                self.data = FunctionsLibrary.model_validate_json(file.read())
        except ValidationError:
            print(f"{file_name} has invalid json format.")
            sys.exit(1)
        except FileNotFoundError:
            print(f"{file_name} is missing.")
            sys.exit(1)

        self.prompt = ""

    def add_user_prompt(self, prompt: str) -> str:
        """
        Appends the user prompt to the ongoing system prompt sequence.

        Args:
            prompt (str): The raw text provided by the user.

        Returns:
            str: The fully constructed prompt for generating an answer.
        """
        self.prompt += (f"<|im_start|>user\n{prompt}\n<|im_end|>" +
                        "\n<|im_start|>assistant\n<think>\n</think>\n")

        return self.prompt

    @staticmethod
    def get_prompts_from_json(file_path: str) -> Any:
        """
        Loads user prompts from a JSON file.

        Args:
            file_path (str): The path to the file containing prompts.

        Returns:
            Any: The parsed PromptLibrary object.
        """
        try:
            with open(file_path, "r") as file:
                return PromptLibrary.model_validate_json(file.read())
        except ValidationError:
            print(f"{file_path} has invalid json format.")
            sys.exit(1)
        except FileNotFoundError:
            print(f"{file_path} is missing.")
            sys.exit(1)

    @staticmethod
    def _get_parameters(params: Dict[str, ParameterType]) -> str:
        """
        Formats a dictionary of parameters into a human-readable string.

        Args:
            params (Dict[str, ParameterType]): The parameters to format.

        Returns:
            str: A formatted string describing the parameters.
        """
        params_str = ""

        for name, param in params.items():
            type = param.type
            params_str += f"\t\t- {name}: {type}\n"

        return params_str

    def get_functions_data_to_str(self) -> str:
        """
        Creates a system prompt instructing the LLM on which function
            to pick.

        Returns:
            str: A system prompt containing descriptions of available
                functions.
        """
        self.prompt = \
            """
            <|im_start|>system
            You are a function calling LLM and your task is to pick
            ONLY the function based on the user prompt.
            (Write only the function name)

            Those are the functions that you can pick:
            """

        data_str = ""
        for function in self.data.root:
            data_str += f"Function name: {function.name}\n"
            data_str += f"\t1. Description {function.description}\n"
            data_str += "\t2. Parameters\n"
            data_str += self._get_parameters(function.parameters)
            data_str += \
                f"\t3. Return:\n\t\t- type: {function.returns.type}\n"

        data_str += "<|im_end|>\n"
        self.prompt += data_str

        return self.prompt

    def get_function_data_from_name(self, name: str) -> Tuple[str,
                                                              FunctionModel]:
        """
        Builds a prompt to instruct the LLM to generate arguments for a
            specific function.

        Args:
            name (str): The name of the target function.

        Returns:
            Tuple[str, FunctionModel]: A tuple containing the prompt
                string and the function data model.
        """
        self.prompt = \
            """
<|im_start|>system
You generate function call arguments.

Rules:
- Extract and copy RAW user input values.
- Never execute the function intent.
- Never compute, transform, infer, normalize, or rewrite values.
- Keep original spelling, casing, spacing, and punctuation from user input.
- Return only the argument object content, no explanations.
- Ignore function descriptions entirely while filling arguments.
- Pay attention to the name when using fn_greet

Important behavior:
- If the user asks to reverse "hello", argument must be "hello" (not "olleh").
- If the function is fn_substitute_string_with_regex and user intent is
  numbers, set regex to \\d+ exactly.

Examples:
User: Reverse the string 'hello'
Function: fn_reverse_string
Correct args:
{"s":"hello"}

User: Greet shrek
Function: fn_greet
Correct args:
{"s":"shrek"}

User: Replace numbers in 'abc123' with 'X'
Function: fn_substitute_string_with_regex
Correct args:
{"source_string":"abc123","regex":"\\d+","replacement":"X"}
            """

        data_str = ""

        for function in self.data.root:
            if function.name == name:
                data_str += f"Function name: {function.name}\n"
                data_str += "\t1. Parameters\n"
                data_str += self._get_parameters(function.parameters)
                data_str += \
                    ("\t2. Return:\n\t\t- type: " +
                     f"{function.returns.type}\n")
                func_data = function
                break

        self.prompt = self.prompt + data_str + "<|im_end|>\n"
        return self.prompt, func_data
