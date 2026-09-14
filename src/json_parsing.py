from json import JSONDecodeError, load
from pydantic import ValidationError, TypeAdapter
from .classes import FunctionDef, PromptEntry
from argparse import ArgumentParser, Namespace


def find_target_file() -> Namespace:
    args_parser: ArgumentParser = ArgumentParser()
    args_parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    args_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    return args_parser.parse_args()


def load_function_definitions(
    file_target: str,
) -> list[FunctionDef] | None:
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
    except PermissionError as e:
        print(f"Not permission to read the file {file_target}: {e}")
    except JSONDecodeError as e:
        print(f"File {file_target} is not in json format: {e}")
    except ValidationError as e:
        print(f"File {file_target} is not valide: {e}")


def load_prompts(file_target: str) -> list[PromptEntry] | None:
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
    except PermissionError as e:
        print(f"Not permission to read the file {file_target}: {e}")
    except JSONDecodeError as e:
        print(f"File {file_target} is not in json format: {e}")
    except ValidationError as e:
        print(f"File {file_target} is not valide: {e}")


def parsing() -> tuple[list[FunctionDef] | None, list[PromptEntry] | None]:
    args_parsed: Namespace = find_target_file()
    return (
        load_function_definitions(args_parsed.functions_definition),
        load_prompts(args_parsed.input),
    )
