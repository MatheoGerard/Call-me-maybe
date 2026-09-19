from json import JSONDecodeError, load
from pydantic import ValidationError, TypeAdapter
from .parser_classes import FunctionDef, PromptEntry


def load_function_definitions(
    file_target: str,
) -> list[FunctionDef] | None:
    """
    Load and validate the function definitions from a JSON file.

    Reads the JSON file at the given path and validates its content as
    a list of FunctionDef. Errors are not raised: a message describing
    the problem is printed and None is returned.

    Args:
        file_target: Path to the JSON file containing the function
            definitions.

    Returns:
        The list of validated FunctionDef, or None if the file is not
        found, cannot be read, is not valid JSON, or does not match the
        expected structure.
    """

    try:
        with open(file_target, "r") as file:
            data = load(file)
        adapter: TypeAdapter[list[FunctionDef]] = TypeAdapter(
            list[FunctionDef]
        )
        functions_definitions: list[FunctionDef] = adapter.validate_python(
            data
        )
        return functions_definitions
    except FileNotFoundError as e:
        print(f"File {file_target} not found: {e}")
        return None
    except PermissionError as e:
        print(f"Not permission to read the file {file_target}: {e}")
        return None
    except JSONDecodeError as e:
        print(f"File {file_target} is not in json format: {e}")
        return None
    except ValidationError as e:
        print(f"File {file_target} is not valide: {e}")
        return None


def load_prompts(file_target: str) -> list[PromptEntry] | None:
    """
    Load and validate the prompts from a JSON file.

    Reads the JSON file at the given path and validates its content as
    a list of PromptEntry. Errors are not raised: a message describing
    the problem is printed and None is returned.

    Args:
        file_target: Path to the JSON file containing the prompts.

    Returns:
        The list of validated PromptEntry, or None if the file is not
        found, cannot be read, is not valid JSON, or does not match the
        expected structure.
    """

    try:
        with open(file_target, "r") as file:
            data = load(file)
        adapter: TypeAdapter[list[PromptEntry]] = TypeAdapter(
            list[PromptEntry]
        )
        prompts: list[PromptEntry] = adapter.validate_python(data)
        return prompts
    except FileNotFoundError as e:
        print(f"File {file_target} not found: {e}")
        return None
    except PermissionError as e:
        print(f"Not permission to read the file {file_target}: {e}")
        return None
    except JSONDecodeError as e:
        print(f"File {file_target} is not in json format: {e}")
        return None
    except ValidationError as e:
        print(f"File {file_target} is not valide: {e}")
        return None


def load_vocab(vocab_file_path: str) -> dict[int, str]:
    """
    Load a vocabulary file and invert it into an id-to-token mapping.

    Reads a JSON file that maps each token string to its integer id,
    and returns the reverse mapping, so that a token can be found from
    its id.

    Args:
        vocab_file_path: Path to the JSON vocabulary file, encoded in
            UTF-8. It must contain an object mapping token strings to
            integer ids.

    Returns:
        A dictionary mapping each token id to its token string. If two
        tokens share the same id, only the last one read is kept.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        AttributeError: If the JSON content is not an object.
    """

    with open(vocab_file_path, "r", encoding="utf-8") as f:
        data = load(f)
    return {v: k for k, v in data.items()}
