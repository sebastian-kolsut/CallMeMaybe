from llm_sdk.llm_sdk import Small_LLM_Model
from src.PromptGenerator import PromptGenerator
import numpy as np


class Prompter:
    def __init__(self, model: Small_LLM_Model, gen: PromptGenerator):
        self.model = model
        self.gen = gen

    def _get_function_names_tokens(self):
        tokenized_func_names = []

        for func in self.gen.data:
            tokenized_func_names.append(
                self.model.encode(func['name'])[0].tolist())

        return tokenized_func_names

    @staticmethod
    def _get_i_tokens(i: int, tokenized_func_names, found_name, eos_token):
        tokens = []

        for name in tokenized_func_names:
            if i < len(name) and found_name == name[:i]:
                tokens.append(name[i])

        if not tokens:
            return [eos_token]

        return tokens

    @staticmethod
    def _apply_mask_to_logits(tokens, logits):
        logits_copy = np.array(logits, dtype=float)
        masked = np.full(logits_copy.shape, -np.inf)

        masked[tokens] = logits_copy[tokens]
        masked = masked.tolist()

        return masked

    def get_fn_name_from_llm(self, encoded):
        eos_token = self.model.encode("<|im_end|>")[0].tolist()[0]
        tokenized_func_names = self._get_function_names_tokens()
        found_name = []

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

        return self.model.decode(found_name)

    def _get_number_mask(self):
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

    def _get_argument_from_llm(self, type: str, prompt: str):
        encoded = self.model.encode(prompt)[0].tolist()
        if type == "number":
            tokens = self._get_number_mask()

        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(encoded)
            if type == "number":
                logits = self._apply_mask_to_logits(tokens, logits)
            next_word = int(np.argmax(logits))
            encoded.append(next_word)
            if "\n" in self.model.decode(next_word):
                break

        return encoded

    def get_args_to_function(self, func_data: dict, prompt: str):
        prompt += "{\n"
        for name, type in func_data['parameters'].items():
            prompt += f"\"{name}\":"
            prompt = self.model.decode(
                self._get_argument_from_llm(type['type'], prompt))
            prompt += "\n"

        arguments = prompt.split("{\n")[-1]
        print(arguments)
