from .generator import select_function_name, generate_function
from .json_parsing import parsing, load_vocab
import json
import os
from llm_sdk import Small_LLM_Model


def main() -> None:
    functions_def, prompts = parsing()
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

    # 4. Écriture du fichier JSON de sortie
    # output_dir = os.path.dirname(output_path)
    # if output_dir:
    #   os.makedirs(output_dir, exist_ok=True)


#
#   with open(output_path, "w", encoding="utf-8") as f:
#      json.dump(results, f, indent=2)
#
#   print(f"\nRésultats sauvegardés avec succès dans : {output_path}")


if __name__ == "__main__":
    main()

# def main() -> None:
#   print("LOAD LLM...")
#  llm: Small_LLM_Model = Small_LLM_Model()
#
#   prompt: str = "Q: What is the sum of 2 and 3?\nA:"
#  context_tokens: list[int] = llm.encode(prompt)[0].tolist()
#
#   print(f"Prompt:\n{prompt}")
#  print("Réponse: ", end="", flush=True)
#
#   for _ in range(50):
#      probabilities = llm.get_logits_from_input_ids(context_tokens)
#     next_token: int = int(np.argmax(probabilities))
#
#       context_tokens.append(next_token)
#
#       word: str = llm.decode([next_token])
#      print(word, end="", flush=True)
#
#   print("\n")
