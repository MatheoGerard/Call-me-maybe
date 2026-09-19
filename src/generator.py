import os
from json import dump
import re
import math
from .state_machines import (
    NumberCheck,
    PrefixCheck,
    StringCheck,
    FreeTextCheck,
)
from .parser_classes import FunctionDef, TypeSpec


def dump_result(res: list[dict], output_name: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        with open(output_name, "w", encoding="utf-8") as file:
            dump(res, file, indent=2, ensure_ascii=False)

        print(f"✅ Résultats sauvegardés avec succès dans : {output_name}")
    except PermissionError as e:
        print(
            f"Erreur de permission lors de l'écriture dans {output_name}: {e}"
        )
    except Exception as e:
        print(f"Une erreur est survenue lors de la sauvegarde : {e}")


def find_function_by_name(
    name: str, fun_def: list[FunctionDef]
) -> None | FunctionDef:
    """
    Searches a list of function definitions for the one matching the
    given name, returning None if no match is found.
    """

    for function in fun_def:
        if function.name == name:
            return function
    return None


def create_check(
    param_name: str,
    type_args: TypeSpec,
    original_prompt: str = "",
) -> PrefixCheck | StringCheck | NumberCheck | FreeTextCheck:
    if type_args.type == "number":
        return NumberCheck()
    elif type_args.type == "boolean":
        return PrefixCheck(possibilities=["true", "false"])
    elif "regex" in param_name or "replace" in param_name:
        return FreeTextCheck()
    else:
        return StringCheck(target_prompt=original_prompt)


def get_next_valid_token(
    llm,
    input_ids: list[int],
    vocab: dict[int, str],
    checker,
) -> tuple[int, str]:
    logits = llm.get_logits_from_input_ids(input_ids)

    if isinstance(logits[0], list):
        logits = logits[0]

    best_token_id = -1
    best_logit = -math.inf

    for token_id, token_str in vocab.items():
        clean_token_str = (
            token_str.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        )
        simulated_checker = checker.model_copy(deep=True)
        is_valid = True

        for char in clean_token_str:
            if not simulated_checker.is_valid(char):
                is_valid = False
                break
            simulated_checker.add_to_generated(char)

        if is_valid and logits[token_id] > best_logit:
            best_logit = logits[token_id]
            best_token_id = token_id

    if best_token_id == -1:
        raise RuntimeError(
            f"Aucun token valide trouvé pour la valeur actuelle '{checker.generated}'"
        )

    clean_best = (
        vocab[best_token_id]
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\t")
    )

    return best_token_id, clean_best


def select_function_name(
    fun_def: list[FunctionDef],
    llm,
    vocab: dict[int, str],
    original_prompt: str,
) -> str:
    possible_names = [f.name for f in fun_def]
    fn_checker = PrefixCheck(possibilities=possible_names)

    selection_prompt = (
        f"Available functions: {possible_names}\n"
        f"User input: {original_prompt}\n"
        f"Function name: "
    )
    input_ids = llm.encode(selection_prompt)[0].tolist()

    while not fn_checker.is_complete():
        token_id, token_str = get_next_valid_token(
            llm, input_ids, vocab, fn_checker
        )
        input_ids.append(token_id)
        for char in token_str:
            fn_checker.add_to_generated(char)

    return fn_checker.generated


def parse_raw_value(raw_val: str, param_type: str):
    cleaned = raw_val.strip(" \"'\n\t")

    if param_type == "number":
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            val_str = match.group(0)
            return float(val_str) if "." in val_str else int(val_str)
        return 0

    elif param_type == "boolean":
        return cleaned.lower().startswith("true")

    else:
        return cleaned


def generate_function(
    func_name: str,
    fun_def: list[FunctionDef],
    llm,
    vocab: dict[int, str],
    original_prompt: str,
) -> dict | None:
    target_func = next((f for f in fun_def if f.name == func_name), None)
    if not target_func:
        return None

    parsed_parameters = {}
    param_list = list(target_func.parameters.items())

    for idx, (param_name, param_spec) in enumerate(param_list):
        already_extracted = ""
        if parsed_parameters:
            already_extracted = f"Extracted so far: {parsed_parameters}\n"

        if param_spec.type == "number" or param_spec.type == "integer":
            ordinal = "first" if idx == 0 else "second"
            prompt_text = (
                f"Examples:\n"
                f"Text: Compute square root of 144 -> Result: 144\n"
                f"Text: Add 10 and 20 -> Result: 10\n\n"
                f"Text: {original_prompt}\n"
                f"{already_extracted}"
                f"Task: Copy the {ordinal} raw number for '{param_name}' directly from Text. DO NOT compute or solve any math expression.\n"
                f"Result: "
            )

        elif "source" in param_name:
            prompt_text = (
                f"Text: {original_prompt}\n"
                f"{already_extracted}"
                f"Task: Copy the full text argument inside quotes or the complete sentence for '{param_name}'.\n"
                f"Result: "
            )

        elif "regex" in param_name:
            prompt_text = (
                f"Examples:\n"
                f"Text: Replace all numbers with X -> Pattern: \\d+\n"
                f"Text: Replace all vowels with * -> Pattern: [aeiouAEIOU]\n"
                f"Text: Substitute 'cat' with 'dog' -> Pattern: cat\n\n"
                f"Text: {original_prompt}\n"
                f"{already_extracted}"
                f"Task: Write ONLY the regex pattern or target word to match for '{param_name}'.\n"
                f"Pattern: "
            )

        elif "replace" in param_name:
            prompt_text = (
                f"Examples:\n"
                f"Text: Replace all numbers with NUMBERS -> Replacement: NUMBERS\n"
                f"Text: Replace all vowels with asterisks -> Replacement: *\n"
                f"Text: Substitute 'cat' with 'dog' -> Replacement: dog\n\n"
                f"Text: {original_prompt}\n"
                f"{already_extracted}"
                f"Task: Write ONLY the replacement value for '{param_name}'.\n"
                f"Replacement: "
            )

        else:
            prompt_text = (
                f"Examples:\n"
                f"Text: Greet john -> Output: john\n"
                f"Text: Reverse the string 'hello' -> Output: hello\n\n"
                f"Text: {original_prompt}\n"
                f"{already_extracted}"
                f"Task: Extract ONLY the target entity/value for '{param_name}' (exclude verbs like Greet, Reverse).\n"
                f"Output: "
            )

        current_input_ids = llm.encode(prompt_text)[0].tolist()
        checker = create_check(param_name, param_spec, original_prompt)

        steps = 0
        max_steps = 32

        while not checker.is_complete() and steps < max_steps:
            token_id, token_str = get_next_valid_token(
                llm, current_input_ids, vocab, checker
            )
            current_input_ids.append(token_id)

            for char in token_str:
                checker.add_to_generated(char)

            steps += 1

        parsed_parameters[param_name] = parse_raw_value(
            checker.generated, param_spec.type
        )

    return {
        "name": func_name,
        "parameters": parsed_parameters,
    }
