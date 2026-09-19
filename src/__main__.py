from .generator import select_function_name, generate_function, dump_result
from .parser import parsing
from .json_parsing import load_vocab
from llm_sdk import Small_LLM_Model


def main() -> None:
    functions_def, prompts, output_name = parsing()

    if not functions_def or not prompts:
        print("Erreur lors du chargement des fichiers d'entrée.")
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
