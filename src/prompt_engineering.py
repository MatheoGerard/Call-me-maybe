from .parser_classes import TypeSpec


def find_true_prompt(
    original_prompt: str,
    already_extracted: str,
    param_name: str,
    param_spec: TypeSpec,
    idx: int,
) -> str:
    if param_spec.type == "number" or param_spec.type == "integer":
        ordinal = "first" if idx == 0 else "second"
        return (
            f"Examples:\n"
            f"Text: Compute square root of 144 -> Result: 144\n"
            f"Text: Add 10 and 20 -> Result: 10\n\n"
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            f"Task: Copy the {ordinal} raw number for '{param_name}' directly from Text. DO NOT compute or solve any math expression.\n"
            f"Result: "
        )

    elif "source" in param_name:
        return (
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            f"Task: Copy the full text argument inside quotes or the complete sentence for '{param_name}'.\n"
            f"Result: "
        )

    elif "regex" in param_name:
        return (
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
        return (
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
        return (
            f"Examples:\n"
            f"Text: Greet john -> Output: john\n"
            f"Text: Reverse the string 'hello' -> Output: hello\n\n"
            f"Text: {original_prompt}\n"
            f"{already_extracted}"
            f"Task: Extract ONLY the target entity/value for '{param_name}' (exclude verbs like Greet, Reverse).\n"
            f"Output: "
        )
