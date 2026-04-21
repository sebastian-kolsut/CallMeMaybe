*This project has been created as part of the 42 curriculum by skolsut.*

# CallMeMaybe: Function Calling in LLMs

## Description
This project explores the implementation of constrained decoding and function calling mechanisms in Large Language Models (LLMs). The goal is to enforce the output of the model to strictly follow predefined JSON schemas required to execute specific functions. By leveraging token masking and generation control, the project demonstrates how an LLM can be guided to emit valid and accurate tool-calling parameters in a deterministic manner.

## Instructions

### Prerequisites
- Python 3.10+
- `make`

### Installation
You can install the dependencies and prepare the virtual environment using the provided `Makefile`:
```bash
make install
```

### Execution
To run the main script and test the function calling capabilities:
```bash
make run
```

### Debugging & Cleanup
- To run the project in debug mode (via `pdb`), use `make debug`.
- To clean up temporary files and caches (`__pycache__`, `.mypy_cache`, and `.venv`), run `make clean`.
- To execute static analysis and linting (`flake8` and `mypy`), run `make lint`.

## Algorithm Explanation
The constrained decoding approach implemented in this project relies on real-time logit processing during the text generation phase. By analyzing the desired JSON schema for the function arguments, the algorithm creates dynamic token masks. At each generation step, the model's output distribution is filtered to allow only tokens that lead to valid JSON syntax matching the expected structure. This ensures that the generated parameters respect both the data types and the required fields defined in the tool schemas.

## Design Decisions
- **Modularity:** The project is separated into distinct modules (`OutputGenerator`, `Prompter`, `PromptGenerator`, `InputModels`, `OutputModels`) to decouple the logic for prompt construction, API interaction, text generation, and JSON parsing.
- **Strict Typing:** Data validation and type hinting are extensively used to conform to `mypy` strict checks, ensuring robust internal data structures.
- **Logit Warping:** Rather than relying solely on prompt engineering, the implementation intercepts the generation loop to mask out invalid tokens, which significantly increases the reliability of function calling.

## Performance Analysis
- **Accuracy:** The constrained decoding approach guarantees structural correctness for the generated JSON, virtually eliminating parsing errors.
- **Speed:** The overhead added by the logit processor is minimal compared to the token generation time of the LLM, resulting in mostly unaffected throughput.
- **Reliability:** The solution is highly reliable in producing output that correctly maps to function signatures, even with complex schemas involving enums or nested objects.

## Challenges Faced
- **Tokenization Quirks:** Handling the intricacies of BPE tokenization, where parts of a word or JSON syntax might be merged into unexpected tokens, required careful manipulation of the allowable token space.
- **Context Window Management:** Formulating the system prompt to accurately represent multiple available functions without overwhelming the model's context or confusing its generation.
- **Strict Linting Requirements:** Integrating type annotations and silencing `mypy` without resorting to excessive `Any` fallbacks required substantial refactoring of previously implicit typing.

## Testing Strategy
The implementation was validated using a structured set of test cases containing various function definitions (`functions_definition.json`) and complex querying scenarios. The outputs were verified against expected parsed JSON representations (`function_calling_results.json`) to confirm that all parameters were correctly identified, typed, and formatted. The project was further validated through continuous `flake8` and `mypy` checks.

## Example Usage
Once the environment is set up, running the program will process the input dataset and perform function calls based on the provided schemas.
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```
The output, representing the extracted tool calls and their arguments, will be saved or displayed according to the configured `OutputGenerator`. For example, it might output:
```json
{
  "name": "get_weather",
  "arguments": {
    "location": "Paris",
    "unit": "celsius"
  }
}
```

## Resources
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [JSON Schema Specification](https://json-schema.org/)
- **AI Usage:** Copilot (Gemini 3.1 Pro) was used to refactor type hinting to pass strict `mypy` checks, generate `Makefile` rules, add PEP 257 docstrings to the source code, and compose the structure of this README file.