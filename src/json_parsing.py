import sys
from json import JSONDecodeError, load
from pydantic import ValidationError
from classes import FunctionDef
from pydantic import TypeAdapter


def find_target_file(args: list[str]) -> str | None:
    if "--functions_definition" in args:
        return args[args.index("--functions_definition") + 1]
    return None


def load_function_definitions(
    file_target: str | None,
) -> list[FunctionDef] | None:
    target_name: str

    if not file_target:
        target_name = "data/input/functions_definition.json"
    else:
        target_name = file_target

    try:
        with open(target_name, "r") as file:
            data = load(file)
        adapter: TypeAdapter[list[FunctionDef]] = TypeAdapter(
            list[FunctionDef]
        )
        functions_definitions: list[FunctionDef] = adapter.validate_python(
            data
        )
        return functions_definitions
    except FileNotFoundError as e:
        print(f"File {target_name} not found: {e}")
    except PermissionError as e:
        print(f"Not permission to read the file {target_name}: {e}")
    except JSONDecodeError as e:
        print(f"File {target_name} is not in json format: {e}")
    except ValidationError as e:
        print(f"File {target_name} is not valide: {e}")


if __name__ == "__main__":
    file_name: str | None = find_target_file(sys.argv)
    print(load_function_definitions(file_name))
