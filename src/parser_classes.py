from pydantic import BaseModel
from typing import Literal


class TypeSpec(BaseModel):
    """Specification of a primitive JSON Schema type.

    Describes the type of a single value (for example a function parameter
    or a return value) using one of the primitive type names defined by
    JSON Schema.

    Attributes:
        type: Name of the primitive type. Must be one of ``"string"``,
            ``"boolean"``, ``"number"`` or ``"integer"``.

    Raises:
        pydantic.ValidationError: If ``type`` is not one of the allowed
            values.
    """

    type: Literal["string", "boolean", "number", "integer"]


class FunctionDef(BaseModel):
    """
    Definition of a callable function exposed to the model.

    Describes a function by its name, a human-readable description, the
    types of its parameters and the type of its return value. This is the
    structure used to specify which functions the model may call.

    Attributes:
        name: Unique name of the function.
        description: Human-readable explanation of what the function does.
        parameters: Mapping from each parameter name to its type
            specification.
        returns: Type specification of the value returned by the function.

    Raises:
        pydantic.ValidationError: If a required field is missing or has an
            invalid type.
    """

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class PromptEntry(BaseModel):
    """
    A single natural-language prompt submitted to the model.

    Represents one input entry, typically a user request from which the
    model must select a function and generate its arguments.

    Attributes:
        prompt: Natural-language text of the request.

    Raises:
        pydantic.ValidationError: If ``prompt`` is missing or is not a
            string.
    """

    prompt: str
