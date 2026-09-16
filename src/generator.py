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
) -> PrefixCheck | StringCheck | NumberCheck:
    """
    Choose the correct check engine for a type in entry
    """

    if type_args.type == "number":
        return NumberCheck()
    elif type_args.type == "boolean":
        return PrefixCheck(possibilities=["true", "false"])
    else:
        return StringCheck()


def get_next_valid_token(
    llm,
    input_ids: list[int],
    vocab: dict[int, str],
    checker,
) -> tuple[int, str]:
    """
    Récupère les logits du LLM pour la séquence d'input_ids et sélectionne
    le token valide qui a le logit le plus élevé.
    """
    logits = llm.get_logits_from_input_ids(input_ids)

    # Si le SDK renvoie une liste 2D (ex: batch_size=1), on prend le premier élément
    if isinstance(logits[0], list):
        logits = logits[0]

    best_token_id = -1
    best_logit = -math.inf

    for token_id, token_str in vocab.items():
        # Sauvegarde de l'état textuel avant de tester
        saved_generated = checker.generated
        is_valid = True

        for char in token_str:
            if not checker.is_valid(char):
                is_valid = False
                break
            checker.add_to_generated(char)

        # Restauration de l'état du checker
        checker.generated = saved_generated

        if is_valid and logits[token_id] > best_logit:
            best_logit = logits[token_id]
            best_token_id = token_id

    if best_token_id == -1:
        raise RuntimeError(
            f"Aucun token valide trouvé pour la valeur actuelle '{checker.generated}'"
        )

    return best_token_id, vocab[best_token_id]


def select_function_name(
    fun_def: list[FunctionDef],
    llm,
    vocab: dict[int, str],
    original_prompt: str,
) -> str:
    possible_names = [f.name for f in fun_def]
    fn_checker = PrefixCheck(possibilities=possible_names)

    # Prompt structuré pour orienter le LLM vers le nom de la fonction
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
    """Nettoie et typpe la valeur générée par le LLM."""
    cleaned = raw_val.strip()
    if param_type == "number":
        return float(cleaned) if "." in cleaned else int(cleaned)
    elif param_type == "boolean":
        return cleaned.lower() == "true"
    else:  # string
        # Supprime les guillemets éventuels autour de la valeur
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1]
        return cleaned


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

    for param_name, param_spec in function_find.parameters.items():
        # Construction d'un prompt d'extraction ciblé par paramètre
        extraction_prompt = (
            f"Context: {original_prompt}\n"
            f"Function: {name}\n"
            f"Extract parameter '{param_name}' ({param_spec.type}): "
        )
        input_ids = llm.encode(extraction_prompt)[0].tolist()

        checker = create_check(param_spec)

        while not checker.is_complete():
            token_id, token_str = get_next_valid_token(
                llm, input_ids, vocab, checker
            )
            input_ids.append(token_id)
            for char in token_str:
                checker.add_to_generated(char)

        parsed_parameters[param_name] = parse_raw_value(
            checker.generated, param_spec.type
        )

    return {
        "name": function_find.name,
        "parameters": parsed_parameters,
    }
