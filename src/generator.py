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


def generate_function(
    name: str, fun_def: list[FunctionDef]
) -> PrefixCheck | StringCheck | NumberCheck | None:
    function_find: FunctionDef | None = find_function_by_name(name, fun_def)

    if not function_find:
        return None
