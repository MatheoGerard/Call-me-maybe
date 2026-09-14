# from llm_sdk import Small_LLM_Model
# import numpy as np
from .classes import FunctionDef, PromptEntry
from .json_parsing import parsing
import sys

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

if __name__ == "__main__":
    data: tuple[list[FunctionDef] | None, list[PromptEntry] | None] = parsing()
    if data[0] is None or data[1] is None:
        sys.exit("Program termination due to a missing input file")
    for d in data[0]:
        print(d)
    for d in data[1]:
        print(d)
