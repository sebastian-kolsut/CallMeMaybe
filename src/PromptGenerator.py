from typing import Dict, Tuple
import json


class PromptGenerator:
    def __init__(self, file_name: str) -> None:
        with open(file_name, "r") as file:
            self.data = json.load(file)

        self.prompt = ""

    def add_user_prompt(self, prompt: str) -> str:
        self.prompt += (f"<|im_start|>user\n{prompt}\n<|im_end|>" +
                        "\n<|im_start|>assistant\n<think>\n</think>\n")

        return self.prompt

    @staticmethod
    def _get_parameters(params: Dict[str, Dict[str, str]]) -> str:
        params_str = ""

        for name, param in params.items():
            type = param['type']
            params_str += f"\t\t- {name}: {type}\n"

        return params_str

    def get_functions_data_to_str(self):
        self.prompt = \
            """
            <|im_start|>system
            You are a function calling LLM and your task is to pick
            ONLY the function based on the user prompt.
            (Write only the function name)

            Those are the functions that you can pick:
            """

        data_str = ""
        for function in self.data:
            data_str += f"Function name: {function['name']}\n"
            data_str += f"\t1. Description {function['description']}\n"
            data_str += "\t2. Parameters\n"
            data_str += self._get_parameters(function['parameters'])
            data_str += \
                f"\t3. Return:\n\t\t- type: {function['returns']['type']}\n"

        data_str += "<|im_end|>\n"
        self.prompt += data_str
        return self.prompt

    def get_function_data_from_name(self, name: str) -> Tuple[str, dict]:
        self.prompt = \
            """
            <|im_start|>system
            You are a function calling LLM and your task is to pick
            the arguments from the user's prompt wich will be used
            in the given function. If you finish the given argument write
            a newline.

            This is the function mentioned above:
            """

        data_str = ""

        for function in self.data:
            if function['name'] == name:
                data_str += f"Function name: {function['name']}\n"
                data_str += f"\t1. Description {function['description']}\n"
                data_str += "\t2. Parameters\n"
                data_str += self._get_parameters(function['parameters'])
                data_str += \
                    ("\t3. Return:\n\t\t- type: " +
                     f"{function['returns']['type']}\n")
                func_data = function
                break

        self.prompt = self.prompt + data_str + "<|im_end|>\n"
        return self.prompt, func_data
