from .OutputModels import FunctionCallsLibrary, FunctionCall
from typing import Dict, Any
import os


class OutputGenerator:
    """
    Handles the generation and formatting of output data.

    This class manages a library of function calls and provides
    methods to extract arguments from raw strings, add function
    calls to the output, and save the output to a JSON file.
    """
    def __init__(self, function_data: Any) -> None:
        """
        Initialize the OutputGenerator.

        Args:
            function_data (Any): the functions library data containing
            definitions.
        """
        self.function_calls = FunctionCallsLibrary([])
        self.function_data = function_data

    def create_json_output(self, output_file_name: str) -> None:
        """
        Write the stored function calls to a JSON file.

        Args:
            output_file_name (str): The name and path of the output JSON file.
        """
        if not os.path.exists("data/output"):
            os.makedirs("data/output")

        output_str = self.function_calls.model_dump_json(indent=2)

        with open(output_file_name, "w") as file:
            file.write(output_str)

    def add_function_to_output(self, name: str, args: str,
                               prompt: str) -> None:
        """
        Add a parsed function call to the output library.

        Args:
            name (str): The name of the function called.
            args (str): A string representation of the arguments.
            prompt (str): The prompt that triggered the call.
        """
        self.function_calls.root.append(
            FunctionCall(
                prompt=prompt,
                name=name,
                parameters=self._extract_arguments(args, name))
            )

    def _extract_arguments(self, args: str,
                           func_name: str) -> Dict[str, float | str]:
        """
        Parses a raw argument string and converts values to their proper types.

        Args:
            args (str): The raw string of arguments.
            func_name (str): The name of the target function.

        Returns:
            Dict[str, float | str]: A dictionary of typed arguments.
        """
        arguments_list = args.split("\n")[:-1]

        arguments_dict: Dict[str, float | str] = {}
        for arg in arguments_list:
            arg_name, arg_value = arg.split(": ")
            arg_name = arg_name.strip("\"\' ,")
            arg_value = arg_value.strip("\"\' ,")
            if "\\\\" in arg_value:
                arg_value = arg_value.replace("\\\\", "\\")

            type = self._get_argument_type(arg_name, func_name)
            if type == "number":
                arguments_dict.update({arg_name: float(arg_value)})
            elif type == "string":
                arguments_dict.update({arg_name: arg_value})
            elif type == "bool":
                arguments_dict.update({arg_name: bool(arg_value)})

        return arguments_dict

    def _get_argument_type(self, arg_name: str, func_name: str) -> Any:
        """
        Retrieves the expected type of an argument for a given function.

        Args:
            arg_name (str): The parameter name.
            func_name (str): The name of the function.

        Returns:
            Any: The type of the parameter as defined in the function library.
        """
        for func in self.function_data.root:
            if func.name == func_name:
                return func.parameters[arg_name].type
