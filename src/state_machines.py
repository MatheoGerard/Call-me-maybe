from pydantic import BaseModel, Field
from string import digits


class PrefixCheck(BaseModel):
    """
    Tracks a generated string against a set of allowed strings.

    Used to constrain generation character by character: it keeps the
    text generated so far and only accepts characters that keep it a
    prefix of at least one allowed possibility.

    Attributes:
        possibilities: Allowed complete strings that the generated text
            must eventually match.
        generated: Text generated so far. Defaults to an empty string.
    """

    possibilities: list[str]
    generated: str = ""

    def next_char(self) -> set[str]:
        """
        Return the characters that may validly follow the generated text.

        A possibility contributes its next character only if the
        generated text is a strict prefix of it. Possibilities already
        equal to the generated text are skipped, as they have no next
        character.

        Returns:
            The set of valid next characters. Empty if no possibility
            can be extended.
        """

        valid_chars: set[str] = set()
        for valid in self.possibilities:
            if valid == self.generated:
                continue
            if valid.startswith(self.generated):
                valid_chars.add(valid[len(self.generated)])
        return valid_chars

    def add_to_generated(self, char: str) -> None:
        """
        Append a character to the generated text.

        No validation is performed. Use is_valid() beforehand to check
        that the character is allowed.

        Args:
            char: Character to append to the generated text.
        """

        self.generated += char

    def is_complete(self) -> bool:
        """
        Check whether the generated text matches a possibility exactly.

        Returns:
            True if the generated text is equal to one of the
            possibilities, False otherwise.
        """

        for possibility in self.possibilities:
            if self.generated == possibility:
                return True
        return False

    def is_valid(self, char: str) -> bool:
        """
        Check whether a character is allowed as the next one.

        Args:
            char: Candidate character to test.

        Returns:
            True if char is in the set returned by next_char(), False
            otherwise.
        """

        return char in self.next_char()


class StringCheck(BaseModel):
    """
    Constrains generated free text to be a substring of a target text.

    Used to constrain the generation of a string argument character by
    character. The generated text must remain a substring of the target
    prompt (case-insensitive, ignoring surrounding whitespace and
    quotes). Generation ends when a closing delimiter (newline, double
    quote or single quote) is received after some content.

    Attributes:
        generated: Text generated so far. Defaults to an empty string.
        target_prompt: Text in which the generated content must be
            found. Defaults to an empty string.
        is_done: Whether generation is finished. Set once a closing
            delimiter is received. Excluded from serialization.
    """

    generated: str = ""
    target_prompt: str = ""
    is_done: bool = Field(default=False, exclude=True)

    def is_valid(self, char: str) -> bool:
        """
        Check whether a character is allowed as the next one.

        The candidate text is the generated text plus char, stripped of
        spaces, quotes, newlines and tabs on both ends. It is accepted
        if it is empty, or if it appears (case-insensitive) as a
        substring of the target prompt.

        Args:
            char: Candidate character to test.

        Returns:
            True if the candidate text is empty or found in the target
            prompt, False otherwise.
        """

        candidate = self.generated + char
        cleaned = candidate.strip(" \"'\n\t")
        if not cleaned:
            return True
        return cleaned.lower() in self.target_prompt.lower()

    def add_to_generated(self, char: str) -> None:
        """
        Append a character to the generated text, or end generation.

        If char is a newline, a double quote or a single quote, and the
        generated text already has content (ignoring surrounding spaces,
        quotes, newlines and tabs), it is treated as a closing delimiter:
        is_done is set to True and char is not appended. Otherwise, char
        is appended to the generated text.

        No validation is performed. Use is_valid() beforehand to check
        that the character is allowed.

        Args:
            char: Character to append to the generated text.
        """

        cleaned = self.generated.strip(" \"'\n\t")
        if char in ("\n", '"', "'") and len(cleaned) >= 1:
            self.is_done = True
            return
        self.generated += char

    def is_complete(self) -> bool:
        """
        Check whether generation is finished.

        Returns:
            True if a closing delimiter has been received, False
            otherwise.
        """

        return self.is_done


class FreeTextCheck(BaseModel):
    """
    Accepts any generated free text until a closing delimiter is received.

    Same interface as the other checkers, but without any constraint on
    the content: every character is valid. Generation ends when a
    closing delimiter (newline, double quote or single quote) is
    received after some content.

    Attributes:
        generated: Text generated so far. Defaults to an empty string.
        is_done: Whether generation is finished. Set once a closing
            delimiter is received. Excluded from serialization.
    """

    generated: str = ""
    is_done: bool = Field(default=False, exclude=True)

    def is_valid(self, char: str) -> bool:
        """
        Check whether a character is allowed as the next one.

        Args:
            char: Candidate character to test. Ignored.

        Returns:
            Always True, as free text has no constraint.
        """

        return True

    def add_to_generated(self, char: str) -> None:
        """
        Append a character to the generated text, or end generation.

        If char is a newline, a double quote or a single quote, and the
        generated text already has content (ignoring surrounding spaces,
        quotes, newlines and tabs), it is treated as a closing delimiter:
        is_done is set to True and char is not appended. Otherwise, char
        is appended to the generated text.

        Args:
            char: Character to append to the generated text.
        """

        cleaned = self.generated.strip(" \"'\n\t")
        if char in ("\n", '"', "'") and len(cleaned) >= 1:
            self.is_done = True
            return
        self.generated += char

    def is_complete(self) -> bool:
        """
        Check whether generation is finished.

        Returns:
            True if a closing delimiter has been received, False
            otherwise.
        """

        return self.is_done


class NumberCheck(BaseModel):
    """
    Constrains generated text to look like a number.

    Used to constrain the generation of a numeric argument character by
    character. The text may start with a minus sign, followed by digits
    and at most one decimal point. Generation ends when a terminator
    (space, newline, comma or closing parenthesis) is received after
    some content.

    Attributes:
        generated: Text generated so far, without terminators. Defaults
            to an empty string.
        is_done: Whether a terminator has been received. Excluded from
            serialization.
    """

    generated: str = ""
    is_done: bool = Field(default=False, exclude=True)

    def is_valid(self, char: str) -> bool:
        """
        Check whether a character is allowed as the next one.

        The accepted characters depend on the text generated so far
        (ignoring surrounding whitespace):
        - empty, or only a minus sign: digits, "-" or a space.
        - a decimal point already present: digits or a terminator.
        - otherwise: digits, ".", or a terminator.

        Args:
            char: Candidate character to test.

        Returns:
            True if char is allowed in the current state, False
            otherwise.
        """

        cleaned = self.generated.strip()

        if not cleaned or cleaned == "-":
            return char in (digits + "- ")

        if "." in cleaned:
            return char in digits or char in (" ", "\n", ",", ")")

        return char in (digits + ". \n,)")

    def add_to_generated(self, char: str) -> None:
        """
        Append a character to the generated text, or end generation.

        If char is a terminator (space, newline, comma or closing
        parenthesis) and the generated text is not empty, is_done is
        set to True and char is not appended. A terminator received
        while the text is empty is ignored. Any other character is
        appended.

        No validation is performed. Use is_valid() beforehand to check
        that the character is allowed.

        Args:
            char: Character to append to the generated text.
        """

        cleaned = self.generated.strip()
        if len(cleaned) > 0 and char in (" ", "\n", ",", ")"):
            self.is_done = True
            return
        if char not in (" ", "\n", ",", ")"):
            self.generated += char

    def is_complete(self) -> bool:
        """
        Check whether a complete number has been generated.

        A lone minus sign or an empty text is never complete, even if a
        terminator has been received.

        Returns:
            True if a terminator has been received after a non-empty
            text other than "-", False otherwise.
        """

        cleaned = self.generated.strip()
        if not cleaned or cleaned == "-":
            return False
        return self.is_done
