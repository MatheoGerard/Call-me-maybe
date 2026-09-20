from .parser_classes import TypeSpec


def find_true_prompt(
    original_prompt: str,
    already_extracted: str,
    param_name: str,
    param_spec: TypeSpec,
    idx: int,
) -> str:
    """
    Build the prompt used to extract the value of one parameter.

    Selects a few-shot prompt template according to the type and the
    name of the parameter. The first matching case is used:
    - "number" or "integer" type: asks to copy the idx-th raw number
      from the text, without computing anything.
    - name containing "source": asks to copy the full text argument.
    - name containing "regex": asks for the regex pattern or the word
      to match.
    - name containing "replace": asks for the replacement value.
    - any other case: asks for the target entity or value, excluding
      action verbs.

    Args:
        original_prompt: Natural-language request from which the value
            is extracted.
        already_extracted: Line describing the values already extracted
            for the previous parameters, ending with a newline. Empty
            string if there are none.
        param_name: Name of the parameter to extract.
        param_spec: Type specification of the parameter.
        idx: Position of the parameter in the function's parameter
            list. Only used to choose between "first" and "second" for
            numeric parameters.

    Returns:
        The prompt to give to the model, ending with the label that the
        generated value should follow.
    """

    if param_spec.type == "number" or param_spec.type == "integer":
        ordinal = "first" if idx == 0 else "second"
        return (
            "Examples:\n"
            "Text: Compute square root of 144 -> Result: 144\n"
            "Text: Add 10 and 20 -> Result: 10\n\n"
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            f"Task: Copy the {ordinal} raw number for '{param_name}' "
            "directly from Text. DO NOT compute or solve any math"
            " expression.\n"
            "Result: "
        )

    elif "source" in param_name:
        return (
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            "Task: Copy the full text argument inside quotes or the "
            f"complete sentence for '{param_name}'.\n"
            "Result: "
        )

    elif "regex" in param_name:
        return (
            "Examples:\n"
            "Text: Replace all numbers with X -> Pattern: \\d+\n"
            "Text: Replace all vowels with * -> Pattern: [aeiouAEIOU]\n"
            "Text: Substitute 'cat' with 'dog' -> Pattern: cat\n\n"
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            "Task: Write ONLY the regex pattern or target word to"
            f" match for '{param_name}'.\n"
            "Pattern: "
        )

    elif "replace" in param_name:
        return (
            "Examples:\n"
            "Text: Replace all numbers with NUMBERS -> Replacement: NUMBERS\n"
            "Text: Replace all vowels with asterisks -> Replacement: *\n"
            "Text: Substitute 'cat' with 'dog' -> Replacement: dog\n\n"
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            f"Task: Write ONLY the replacement value for '{param_name}'.\n"
            "Replacement: "
        )

    else:
        return (
            "Examples:\n"
            "Text: Greet john -> Output: john\n"
            "Text: Reverse the string 'hello' -> Output: hello\n\n"
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            f"Task: Extract ONLY the target entity/value for '{param_name}'"
            " (exclude verbs like Greet, Reverse).\n"
            "Output: "
        )
