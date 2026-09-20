from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
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
from .prompt_engineering import find_true_prompt


def dump_result(res: list[dict], output_name: str) -> None:
    """
    Write the results to a JSON file.

    Creates the parent directories of the output path if they do not
    exist, then writes the results as indented JSON, encoded in UTF-8
    with non-ASCII characters kept as is. Errors are not raised: a
    message describing the problem is printed instead.

    Args:
        res: List of result dictionaries to write. Must be
            JSON-serializable.
        output_name: Path of the output JSON file. An existing file is
            overwritten.
    """

    try:
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        with open(output_name, "w", encoding="utf-8") as file:
            dump(res, file, indent=2, ensure_ascii=False)

        print(f"Results successfully saved in: {output_name}")
    except PermissionError as e:
        print(f"Permission error while writing to {output_name}: {e}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")


def create_check(
    param_name: str,
    type_args: TypeSpec,
    original_prompt: str = "",
) -> PrefixCheck | StringCheck | NumberCheck | FreeTextCheck:
    """
    Create the checker matching a function parameter.

    Selects the checker used to constrain the generation of the value
    of a parameter, according to its type and its name:
    - "number" type: NumberCheck.
    - "boolean" type: PrefixCheck restricted to "true" and "false".
    - name containing "regex" or "replace": FreeTextCheck, since the
      value is not necessarily found in the prompt.
    - any other case: StringCheck, which constrains the value to be
      found in the original prompt.

    Args:
        param_name: Name of the parameter. Only checked for the
            substrings "regex" and "replace".
        type_args: Type specification of the parameter.
        original_prompt: Prompt from which the value must be extracted.
            Only used by StringCheck. Defaults to an empty string.

    Returns:
        A new checker instance suited to the parameter.
    """

    if type_args.type == "number" or type_args.type == "integer":
        return NumberCheck()
    elif type_args.type == "boolean":
        return PrefixCheck(possibilities=["true", "false"])
    elif "regex" in param_name or "replace" in param_name:
        return FreeTextCheck()
    else:
        return StringCheck(target_prompt=original_prompt)


def get_next_valid_token(
    llm: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[int, str],
    checker: NumberCheck | StringCheck | PrefixCheck | FreeTextCheck,
) -> tuple[int, str]:
    """
    Select the most likely token that satisfies the checker.

    Gets the logits for the next token from the model, then goes
    through the whole vocabulary. Each token is decoded (byte-level BPE
    markers are replaced by their real characters) and tested
    character by character on a copy of the checker. Among the tokens
    accepted by the checker, the one with the highest logit is
    returned. The checker passed as argument is not modified.

    Args:
        llm: Model used to compute the logits of the next token.
        input_ids: Ids of the tokens generated so far, used as context.
        vocab: Mapping from each token id to its token string.
        checker: Checker constraining the value being generated. Each
            candidate token is simulated on a deep copy of it.

    Returns:
        A tuple of two elements:
        - The id of the best valid token.
        - Its decoded string, where "Ġ" is replaced by a space, "Ċ" by
          a newline and "ĉ" by a tab.

    Raises:
        RuntimeError: If no token of the vocabulary is accepted by the
            checker.
    """

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
            "No valid token found for the current value:"
            f" '{checker.generated}'"
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
    llm: Small_LLM_Model,
    vocab: dict[int, str],
    original_prompt: str,
) -> str:
    """
    Choose the function matching a prompt, token by token.

    Builds a selection prompt listing the available function names and
    the user input, then generates the function name with the model.
    Generation is constrained by a PrefixCheck so that the result is
    always exactly one of the available names. Tokens are added until
    the generated text is a complete name.

    Args:
        fun_def: Definitions of the available functions. Only their
            names are used.
        llm: Model used to generate the name.
        vocab: Mapping from each token id to its token string.
        original_prompt: Natural-language request from which the
            function must be selected.

    Returns:
        The name of the selected function, guaranteed to be one of the
        names in fun_def.

    Raises:
        RuntimeError: If no token of the vocabulary is accepted by the
            checker (raised by get_next_valid_token).
    """

    possible_names = [f.name for f in fun_def]
    fn_checker: PrefixCheck = PrefixCheck(possibilities=possible_names)

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

    return str(fn_checker.generated)


def parse_raw_value(raw_val: str, param_type: str) -> str | float | int:
    """
    Convert a raw generated string into a value of the expected type.

    Strips surrounding spaces, quotes, newlines and tabs, then converts
    the result according to the parameter type:
    - "number": the first number found in the text, as an int if it has
      no decimal point, or a float otherwise. Returns 0 if no number
      is found.
    - "boolean": True if the text starts with "true" (case-insensitive),
      False otherwise.
    - any other type: the cleaned string.

    Args:
        raw_val: Raw text generated for the parameter.
        param_type: Type name of the parameter, as given by
            TypeSpec.type.

    Returns:
        The converted value: an int or a float for "number", a bool for
        "boolean", or a str for any other type.
    """

    cleaned = raw_val.strip(" \"'\n\t")

    if param_type == "number" or param_type == "integer":
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
    llm: Small_LLM_Model,
    vocab: dict[int, str],
    original_prompt: str,
) -> dict | None:
    """
    Generate the arguments of a function call from a prompt.

    Looks up the definition of the function, then generates the value
    of each of its parameters in order. For each parameter, a prompt is
    built (including the values already extracted), a checker suited to
    the parameter is created, and tokens are generated one by one until
    the checker is complete or the step limit is reached. The raw text
    is then converted to the expected type.

    Args:
        func_name: Name of the function to call.
        fun_def: Definitions of the available functions, in which
            func_name is looked up.
        llm: Model used to generate the values.
        vocab: Mapping from each token id to its token string.
        original_prompt: Natural-language request from which the
            arguments are extracted.

    Returns:
        A dictionary with two keys:
        - "name": The name of the function.
        - "parameters": A dictionary mapping each parameter name to its
          generated value.
        Returns None if func_name is not found in fun_def.

    Raises:
        RuntimeError: If no token of the vocabulary is accepted by the
            checker (raised by get_next_valid_token).
    """

    target_func = next((f for f in fun_def if f.name == func_name), None)
    if not target_func:
        return None

    parsed_parameters: dict[str, str | float | int] = {}
    param_list = list(target_func.parameters.items())

    for idx, (param_name, param_spec) in enumerate(param_list):
        already_extracted = ""
        if parsed_parameters:
            already_extracted = f"Extracted so far: {parsed_parameters}\n"

        prompt_text = find_true_prompt(
            original_prompt, already_extracted, param_name, param_spec, idx
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
