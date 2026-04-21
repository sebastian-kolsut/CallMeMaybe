from llm_sdk.llm_sdk import Small_LLM_Model
from src.PromptGenerator import PromptGenerator
from typing import List, Any
import numpy as np


class Prompter:
    """
    Class responsible for generating prompts and enforcing output constraints.
    """
    def __init__(self, model: Small_LLM_Model, gen: PromptGenerator):
        """
        Initialize the Prompter.

        Args:
            model (Small_LLM_Model): The LLM to be used for inference.
            gen (PromptGenerator): The generator instance with prompt data.
        """
        self.model = model
        self.gen = gen

    def _get_function_names_tokens(self) -> List[List[int]]:
        """
        Retrieves the tokenized names of all available functions.

        Returns:
            List[List[int]]: A list of token sequences for each function name.
        """
        tokenized_func_names = []

        for func in self.gen.data.root:
            tokenized_func_names.append(
                self.model.encode(func.name)[0].tolist())

        return tokenized_func_names

    @staticmethod
    def _get_i_tokens(i: int, tokenized_func_names: List[List[int]],
                      found_name: List[int], eos_token: int) -> List[int]:
        """
        Retrieves valid next tokens based on the matched function name so far.

        Args:
            i (int): The current token index being generated.
            tokenized_func_names (List[List[int]]): Encoded function names.
            found_name (List[int]): The sequence of tokens generated so far.
            eos_token (int): The end-of-sequence token ID.

        Returns:
            List[int]: A list of allowed next tokens.
        """
        tokens = []

        for name in tokenized_func_names:
            if i < len(name) and found_name == name[:i]:
                tokens.append(name[i])

        if not tokens:
            return [eos_token]

        return tokens

    @staticmethod
    def _apply_mask_to_logits(tokens: List[int], logits: List[float]) -> Any:
        """
        Applies a mask to the generated logits, leaving only specified tokens.

        Args:
            tokens (List[int]): The allowed tokens.
            logits (List[float]): Raw model output logits.

        Returns:
            Any: The masked logits as a list.
        """
        logits_copy = np.array(logits, dtype=float)
        masked = np.full(logits_copy.shape, -np.inf)

        masked[tokens] = logits_copy[tokens]
        new_masked = masked.tolist()

        return new_masked

    def get_fn_name_from_llm(self, encoded: List[int]) -> str:
        """
        Forces the LLM to output a valid function name from available
            names.

        Args:
            encoded (List[int]): The initial token sequence representing
                context.

        Returns:
            str: The generated, valid function name.
        """
        eos_token = self.model.encode("<|im_end|>")[0].tolist()[0]
        tokenized_func_names = self._get_function_names_tokens()
        found_name: List[int] = []

        i = 0
        while True:
            logits = self.model.get_logits_from_input_ids(encoded)
            tokens = self._get_i_tokens(
                i, tokenized_func_names, found_name, eos_token)
            logits = self._apply_mask_to_logits(tokens, logits)
            next_word = int(np.argmax(logits))
            encoded.append(next_word)
            found_name.append(next_word)
            if next_word == eos_token:
                break
            i += 1

        return str(self.model.decode(found_name))

    def _get_number_mask(self) -> List[int]:
        """
        Creates a list of token IDs allowing for valid numeric characters.

        Returns:
            List[int]: Allowed tokens representing numbers and their operators.
        """
        base_pieces = [
            " -", "0", "1", "2", "3", "4", "5", "6",
            "7", "8", "9", ".", "\n", "-", " "
        ]

        candidate_pieces = base_pieces
        encoded_digits = []

        for piece in candidate_pieces:
            piece_tokens = self.model.encode(piece)[0].tolist()
            encoded_digits.extend(piece_tokens)

        return list(encoded_digits)

    def _get_bool_mask(self) -> List[int]:
        """
        Creates a list of token IDs allowing for valid boolean values.

        Returns:
            List[int]: Allowed tokens encoding "True" and "False".
        """
        base_pieces = ["True", "False"]

        candidate_pieces = base_pieces
        encoded_bool = []

        for piece in candidate_pieces:
            piece_tokens = self.model.encode(piece)[0].tolist()
            encoded_bool.extend(piece_tokens)

        return list(encoded_bool)

    def _get_argument_from_llm(self, type: str, prompt: str) -> Any:
        """
        Forces LLM to generate an argument matching exactly the desired type.

        Args:
            type (str): The expected data type ("number", "bool" or other).
            prompt (str): The existing prompt up to the current argument value.

        Returns:
            Any: The generated token sequence containing the argument value.
        """
        encoded = self.model.encode(prompt)[0].tolist()

        if type == "number":
            tokens = self._get_number_mask()
        elif type == "bool":
            tokens = self._get_bool_mask()

        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(encoded)
            if type == "number" or type == "bool":
                logits = self._apply_mask_to_logits(tokens, logits)
            next_word = int(np.argmax(logits))
            encoded.append(next_word)
            if "\n" in self.model.decode([next_word]):
                break

        return encoded

    def get_args_to_function(self, func_data: Any, prompt: str) -> str:
        """
        Generates arguments for a function, forcing the LLM to follow the
            expected types.

        Args:
            func_data (Any): The definition data of the target function.
            prompt (str): The initial prompt string for the LLM.

        Returns:
            str: A multi-line string containing the generated parameters
                and their values.
        """
        prompt += "{\n"
        for name, type in func_data.parameters.items():
            prompt += f"\"{name}\":"
            prompt = self.model.decode(
                self._get_argument_from_llm(type.type, prompt))

        arguments = prompt.split("{\n")[-1]
        return arguments
