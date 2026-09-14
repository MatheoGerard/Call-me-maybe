from pydantic import BaseModel
from typing import Literal
from string import digits


class TypeSpec(BaseModel):
    """
    Represents the expected type of a function parameter or return value,
    as declared in functions_definition.json.
    """

    type: Literal["string", "boolean", "number"]


class FunctionDef(BaseModel):
    """
    Represents the expected format of a function definition in functions_definition.json.
    """

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class PromptEntry(BaseModel):
    """
    Represents a single natural language prompt to be processed by the
    function calling system, as declared in function_calling_tests.json.
    """

    prompt: str


class PrefixCheck(BaseModel):
    """
    Represents a automation to find the next character to add.
    """

    possibilities: list[str]
    generated: str = ""

    def next_char(self) -> set[str]:
        valid_chars: set[str] = set()
        for valid in self.possibilities:
            if valid == self.generated:
                continue
            if valid.startswith(self.generated):
                valid_chars.add(valid[len(self.generated)])
        return valid_chars

    def add_to_generated(self, char: str) -> None:
        self.generated += char

    def is_complete(self) -> bool:
        for possibility in self.possibilities:
            if self.generated == possibility:
                return True
        return False


# class StringCheck(BaseModel):
#   """
#  Represents a automation to find the next character to add.
# """

# generated: str = ""

# def


class NumberCheck(BaseModel):
    """
    Represents a automation to find the next character to add.
    """

    generated: str = ""

    def clean_digits(self) -> str:
        return self.generated.lstrip("-")

    def next_char(self) -> set[str]:
        if not self.generated:
            return set(digits + "-")
        elif self.generated == "-":
            return set(digits)
        elif self.clean_digits() == "0":
            return set(".")
        elif self.generated[-1] == ".":
            return set(digits)
        elif "." in self.generated:
            return set(digits)
        else:
            return set(digits + ".")

    def is_complete(self) -> bool:
        if not self.generated or self.generated == "-":
            return False
        if self.generated[-1] == ".":
            return False
        return True
