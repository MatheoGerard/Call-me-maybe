from pydantic import BaseModel, Field
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

    def is_valid(self, char: str) -> bool:
        return char in self.next_char()


class StringCheck(BaseModel):
    generated: str = ""
    target_prompt: str = ""
    is_done: bool = Field(default=False, exclude=True)

    def is_valid(self, char: str) -> bool:
        candidate = self.generated + char
        cleaned_candidate = candidate.strip(" \"'\n\t")
        if not cleaned_candidate:
            return True
        # Seuls les caractères formant une sous-chaîne du prompt sont valides
        return cleaned_candidate.lower() in self.target_prompt.lower()

    def add_to_generated(self, char: str) -> None:
        if char in ('"', "'", "\n") and len(self.generated.strip()) > 0:
            self.is_done = True
            return
        self.generated += char

    def is_complete(self) -> bool:
        cleaned = self.generated.strip(" \"'\n\t")
        if cleaned.lower() in self.target_prompt.lower() and len(cleaned) >= 2:
            return self.is_done
        return self.is_done


class NumberCheck(BaseModel):
    generated: str = ""
    is_done: bool = Field(default=False, exclude=True)

    def is_valid(self, char: str) -> bool:
        cleaned = self.generated.strip()

        if not cleaned or cleaned == "-":
            return char in (digits + "- ")

        if "." in cleaned:
            return char in digits or char in (" ", "\n", ",", ")")

        # Autorise les chiffres ou le séparateur de fin
        return char in (digits + ". \n,)")

    def add_to_generated(self, char: str) -> None:
        cleaned = self.generated.strip()
        # On ne passe à True que si on a déjà au moins un chiffre ET qu'on lit un espace/séparateur
        if len(cleaned) > 0 and char in (" ", "\n", ",", ")"):
            self.is_done = True
            return
        if char not in (" ", "\n", ",", ")"):
            self.generated += char

    def is_complete(self) -> bool:
        cleaned = self.generated.strip()
        if not cleaned or cleaned == "-":
            return False
        return self.is_done
