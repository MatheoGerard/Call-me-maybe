from pydantic import BaseModel
from typing import Literal


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
