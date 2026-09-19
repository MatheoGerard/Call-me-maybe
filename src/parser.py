from argparse import ArgumentParser, Namespace
from .parser_classes import FunctionDef, PromptEntry
from .json_parsing import load_function_definitions, load_prompts


def load_arguments() -> Namespace:
    """
    Parse the command-line arguments of the program.

    Defines three optional arguments, each with a default path:
    --functions_definition, --input and --output.

    Returns:
        A Namespace with the following attributes:
        - functions_definition: Path to the JSON file describing the
          available functions. Defaults to
          "data/input/functions_definition.json".
        - input: Path to the JSON file containing the prompts to
          process. Defaults to
          "data/input/function_calling_tests.json".
        - output: Path to the JSON file where results are written.
          Defaults to "data/output/output.json".

    Raises:
        SystemExit: If an unknown argument is given, or if --help is
            requested (raised by argparse).
    """

    args_parser: ArgumentParser = ArgumentParser()
    args_parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    args_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    args_parser.add_argument(
        "--output",
        default="data/output/output.json",
    )
    return args_parser.parse_args()


def parsing() -> (
    tuple[list[FunctionDef] | None, list[PromptEntry] | None, str]
):
    """
    Read the command-line arguments and load the input files.

    Parses the command-line arguments, then loads the function
    definitions and the prompts from the paths they specify. Loading
    errors are not raised: they are printed by the loading functions
    and the corresponding value is None.

    Returns:
        A tuple of three elements:
        - The list of validated FunctionDef, or None if loading the
          functions definition file failed.
        - The list of validated PromptEntry, or None if loading the
          input file failed.
        - The path of the output file.
    """

    args_parsed: Namespace = load_arguments()
    return (
        load_function_definitions(args_parsed.functions_definition),
        load_prompts(args_parsed.input),
        args_parsed.output,
    )
