from llm_sdk.llm_sdk import Small_LLM_Model
from src.PromptGenerator import PromptGenerator
from src.Prompter import Prompter
from .OutputGenerator import OutputGenerator
from typing import List, Tuple
import sys
import os


def parse(argv: List[str]) -> Tuple[str, str, str]:
    """
    Parses command-line arguments to determine input and output file paths.

    Args:
        argv (List[str]): The arguments provided when running the script.

    Returns:
        Tuple[str, str, str]: Paths to the function definitions, input user
            prompts, and output JSON file.
    """
    func_definitions = "data/input/functions_definition.json"
    input_file = "data/input/function_calling_tests.json"
    output_file = "data/output/function_calling_results.json"

    if "--functions_definition" in argv:
        i = argv.index("--functions_definition") + 1
        func_definitions = "data/input/" + argv[i]
    if "--input" in argv:
        i = argv.index("--input") + 1
        input_file = "data/input/" + argv[i]
    if "--output" in argv:
        i = argv.index("--output") + 1
        output_file = "data/output/" + argv[i]

    if not (func_definitions.endswith(".json") and
            input_file.endswith(".json") and
            output_file.endswith(".json")):
        print("Invalid file format. (*.json required)")
        sys.exit(1)

    return func_definitions, input_file, output_file


def main() -> None:
    """
    Main execution pipeline for the CallMeMaybe function calling application.
    """
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    func_definitions, input_file, output_file = parse(sys.argv)

    model = Small_LLM_Model()
    gen = PromptGenerator(func_definitions)
    prompter = Prompter(model, gen)
    output_generator = OutputGenerator(gen.data)

    prompts = gen.get_prompts_from_json(input_file)

    for user_prompt in prompts.root:
        prompt = gen.get_functions_data_to_str()
        prompt = gen.add_user_prompt(user_prompt.prompt)

        encoded = model.encode(prompt)[0].tolist()
        name = prompter.get_fn_name_from_llm(encoded)

        args_prompt, func_data = gen.get_function_data_from_name(name)
        args_prompt = gen.add_user_prompt(user_prompt.prompt)
        args = prompter.get_args_to_function(func_data, args_prompt)

        output_generator.add_function_to_output(name, args, user_prompt.prompt)

    output_generator.create_json_output(output_file)


if __name__ == "__main__":
    main()
