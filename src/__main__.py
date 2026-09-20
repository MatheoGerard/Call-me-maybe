from .generator import select_function_name, generate_function, dump_result
from .parser import parsing
from .json_parsing import load_vocab
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


def main() -> None:
    """
    Run the function-calling pipeline on all the input prompts.

    Loads the function definitions and the prompts from the files given
    on the command line, then loads the model and its vocabulary. For
    each prompt, selects the matching function, generates its
    arguments, and prints the result. Once all prompts are processed,
    the results are written to the output file. If an input file could
    not be loaded, an error message is printed and the function returns
    without doing anything else.

    Each entry written to the output file is a dictionary with three
    keys: "prompt", "name" and "parameters".

    Raises:
        RuntimeError: If no token of the vocabulary is accepted by a
            checker during generation (raised by get_next_valid_token).
    """

    functions_def, prompts, output_name = parsing()

    if not functions_def or not prompts:
        print("Error loading input files.")
        return

    print("Loading LLM...", flush=True)
    llm = Small_LLM_Model()
    print("LLM ready!", flush=True)

    vocab_path = llm.get_path_to_vocab_file()
    vocab = load_vocab(vocab_path)

    results = []

    for i, entry in enumerate(prompts):
        prompt_text = entry.prompt

        selected_name = select_function_name(
            functions_def, llm, vocab, prompt_text
        )

        func_call = generate_function(
            selected_name, functions_def, llm, vocab, prompt_text
        )

        if func_call:
            result_item = {
                "prompt": prompt_text,
                "name": func_call["name"],
                "parameters": func_call["parameters"],
            }
            results.append(result_item)
            print(
                f"[{i + 1}/{len(prompts)}] Generated:", result_item, flush=True
            )
    dump_result(results, output_name)


if __name__ == "__main__":
    main()
