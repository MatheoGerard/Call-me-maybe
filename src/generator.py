import re
import math
from .classes import (
    FunctionDef,
    NumberCheck,
    PrefixCheck,
    StringCheck,
    TypeSpec,
)


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
    type_args: TypeSpec,
    original_prompt: str = "",
) -> PrefixCheck | StringCheck | NumberCheck:
    if type_args.type == "number":
        return NumberCheck()
    elif type_args.type == "boolean":
        return PrefixCheck(possibilities=["true", "false"])
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
        # On crée un clone ou un état simulé du checker pour ce token
        simulated_checker = checker.model_copy(deep=True)
        is_valid = True

        for char in clean_token_str:
            if not simulated_checker.is_valid(char):
                is_valid = False
                break
            # On applique le caractère sur le clone pour mettre à jour son état interne (ex: transition d'automate)
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
    cleaned = raw_val.strip()

    if param_type == "number":
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            val_str = match.group(0)
            return float(val_str) if "." in val_str else int(val_str)
        return 0

    elif param_type == "boolean":
        return cleaned.lower().startswith("true")

    else:  # string
        # Garde seulement la 1ere ligne et retire guillemets/espaces autour
        cleaned = cleaned.split("\n")[0]
        return cleaned.strip(" \"'\t")


def generate_function(
    name: str,
    fun_def: list[FunctionDef],
    llm,
    vocab: dict[int, str],
    original_prompt: str,
) -> dict | None:
    function_find = next((f for f in fun_def if f.name == name), None)
    if not function_find:
        return None

    parsed_parameters: dict = {}
    param_list = list(function_find.parameters.items())

    for idx, (param_name, param_spec) in enumerate(param_list):
        already_extracted = ""
        if parsed_parameters:
            already_extracted = (
                f"Given extracted parameters: {parsed_parameters}\n"
            )

        if param_spec.type == "number":
            ordinal = "first" if idx == 0 else "second"
            prompt_text = (
                f"Input text: {original_prompt}\n"
                f"{already_extracted}"
                f"Extract the {ordinal} number value for parameter '{param_name}': "
            )
        else:
            prompt_text = (
                f"Text: {original_prompt}\n"
                f"{already_extracted}"
                f"Extract the exact substring for '{param_name}' verbatim from Text.\n"
                f"Substring: "
            )

        current_input_ids = llm.encode(prompt_text)[0].tolist()
        checker = create_check(param_spec, original_prompt)

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
        "name": function_find.name,
        "parameters": parsed_parameters,
    }
