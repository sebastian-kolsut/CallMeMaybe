from llm_sdk.llm_sdk import Small_LLM_Model
from src.PromptGenerator import PromptGenerator
from src.Prompter import Prompter
import time


# EOS Token ID: 151645
def main() -> None:
    model = Small_LLM_Model()
    user_prompt = "WHat is the sum of 200 and 14"

    before = time.time()
    gen = PromptGenerator("data/input/functions_definition.json")
    prompter = Prompter(model, gen)
    prompt = gen.get_functions_data_to_str()
    prompt = gen.add_user_prompt(user_prompt)

    encoded = model.encode(prompt)[0].tolist()

    decoded = prompter.get_fn_name_from_llm(encoded)
    print(decoded)
    args_prompt, func_data = gen.get_function_data_from_name(decoded)
    args_prompt = gen.add_user_prompt(user_prompt)
    prompter.get_args_to_function(func_data, args_prompt)
    print(time.time() - before)


if __name__ == "__main__":
    main()
