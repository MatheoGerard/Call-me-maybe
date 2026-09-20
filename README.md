*This project has been created as part of the 42 curriculum by mgerard.*

# call me maybe

Function calling with a small LLM, made reliable through **constrained decoding**.

## Table of contents

- [Description](#description)
- [Instructions](#instructions)
- [Example usage](#example-usage)
- [Algorithm explanation](#algorithm-explanation)
- [Design decisions](#design-decisions)
- [Performance analysis](#performance-analysis)
- [Challenges faced](#challenges-faced)
- [Testing strategy](#testing-strategy)
- [Project structure](#project-structure)
- [Resources](#resources)

## Description

**call me maybe** is an introduction to *function calling* in Large Language Models. Given a
natural-language request such as *"What is the sum of 2 and 3?"*, the program does not answer the
question. Instead, it translates the request into a structured call to one of the available
functions:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2, "b": 3 }
}
```

The difficulty is that small models (here **Qwen3-0.6B**, about 0.6 billion parameters) are
unreliable when simply prompted to "output JSON": they invent function names, forget parameters,
return strings instead of numbers, or wrap the answer in prose. This project solves the problem by
**constrained decoding**: at every generation step, the tokens that would make the output invalid
are removed from consideration *before* the model picks the next token. The output is therefore
valid by construction, whatever the model would have preferred to say.

### Goal

- Read a list of function definitions and a list of natural-language prompts (JSON files).
- For each prompt, select the right function and extract its arguments with the right types.
- Write all the results to a JSON file that is always syntactically valid and schema-compliant.
- Do it with a tiny model, by controlling the decoding process rather than relying on the model to
  follow instructions.

### Overview

1. The function name is generated under a constraint that only allows the exact names of the
   available functions.
2. Each parameter is then generated one after the other, under a constraint that depends on its
   type (number, boolean, string).
3. The final JSON object is assembled and written by the program itself (never by the LLM), so
   the structure (keys, brackets, quotes, commas) cannot be wrong.

## Instructions

### Requirements

- Python **3.10 or later** (the project was developed with Python 3.14)
- [`uv`](https://docs.astral.sh/uv/) as package and environment manager
- The `llm_sdk` package provided with the subject, placed in `./llm_sdk`
- Enough disk space for the Qwen3-0.6B weights, which are downloaded from Hugging Face on the
  first run

Python dependencies (declared in `pyproject.toml`): `numpy`, `pydantic`, `llm_sdk`.

### Installation

```bash
make install      # runs `uv sync`
```

### Execution

```bash
make run
```

The three input/output paths can be overridden:

```bash
make run \
  FUNC_DEF=data/input/functions_definition.json \
  INPUT=data/input/function_calling_tests.json \
  OUTPUT=data/output/function_calls.json
```

Or directly, without `make`:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

> **Note:** the `run` and `debug` targets of the Makefile set the `HF_HOME` environment variable
> to store the Hugging Face cache outside of the home directory (`sgoinfre` on the 42 machines).
> Edit this path in the Makefile if you run the project elsewhere.

### Command-line arguments

| Argument                 | Default                                   | Description                                     |
| ------------------------ | ----------------------------------------- | ----------------------------------------------- |
| `--functions_definition` | `data/input/functions_definition.json`    | JSON file describing the available functions    |
| `--input`                | `data/input/function_calling_tests.json`  | JSON file containing the prompts to process     |
| `--output`               | `data/output/output.json`                 | JSON file where the results are written         |

The output directory is created automatically if it does not exist.

### Makefile targets

| Target         | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `make install` | Install the dependencies with `uv sync`                                     |
| `make run`     | Run the program                                                             |
| `make debug`   | Run the program under `pdb`                                                 |
| `make lint`    | Run `flake8` and `mypy` (with strict typing flags)                          |
| `make clean`   | Remove `__pycache__`, `*.pyc`, `.mypy_cache` and `.pytest_cache`            |

### Input and output formats

**`functions_definition.json`**: a list of functions.

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

Supported parameter types: `string`, `boolean`, `number`, `integer`.

**`function_calling_tests.json`**: a list of prompts.

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet john" }
]
```

**Output**: a list of results, one per prompt, containing the original prompt, the selected
function name and its parameters.

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  }
]
```

Input files are validated with Pydantic. If a file is missing, unreadable, not valid JSON or does
not match the expected structure, an explicit message is printed and the program stops cleanly
without a traceback.

## Example usage

Default run:

```console
$ make run
Loading LLM...
LLM ready!
[1/3] Generated: {'prompt': 'What is the sum of 2 and 3?', 'name': 'fn_add_numbers', 'parameters': {'a': 2, 'b': 3}}
[2/3] Generated: {'prompt': 'Greet john', 'name': 'fn_greet', 'parameters': {'name': 'john'}}
[3/3] Generated: {'prompt': "Replace all vowels in 'Hello World' with asterisks", 'name': 'fn_substitute_string_with_regex', 'parameters': {'source_string': 'Hello World', 'regex': '[aeiouAEIOU]', 'replacement': '*'}}
Results successfully saved in: data/output/function_calls.json
```

Custom files:

```bash
make run INPUT=my_prompts.json FUNC_DEF=my_functions.json OUTPUT=out/my_results.json
```

Failure handling (missing file):

```console
$ uv run python -m src --input does_not_exist.json
File does_not_exist.json not found: [Errno 2] No such file or directory: 'does_not_exist.json'
Error loading input files.
```

## Algorithm explanation

### Background: how an LLM generates text

An LLM does not output text directly. At each step it receives the sequence of token ids generated
so far and returns one **logit** (a score) per token of its vocabulary, about 150,000 of them for
Qwen3. Normally the token with the highest score (greedy decoding) or a sampled one is appended to
the sequence, and the process repeats.

Constrained decoding intervenes between "the model returns the scores" and "a token is chosen":
every token that would break the desired format is excluded, and the best **remaining** token is
selected.

### Pipeline

```
 prompt ──► [1] select function name ──► [2] for each parameter: extract value ──► [3] build JSON
              PrefixCheck                    NumberCheck / PrefixCheck /             (json.dump,
              (names allowed)                StringCheck / FreeTextCheck              done by code)
```

### Step 1: choosing the function (`select_function_name`)

1. A short prompt is built: the list of available names, the user request, and the beginning of the
   answer (`Function name: `).
2. A `PrefixCheck` is created with the list of valid function names as its only allowed strings.
3. The model is queried step by step. At each step only the tokens that keep the generated text a
   **prefix of at least one function name** are allowed. The best-scoring allowed token is kept.
4. The loop stops when the generated text equals a function name exactly.

The returned name is guaranteed to exist in the function definitions. The model's only freedom is
to *choose among valid names*.

### Step 2: extracting the parameters (`generate_function`)

For each parameter of the selected function, in order:

1. A **few-shot prompt** adapted to the parameter is built (`find_true_prompt`). It contains the
   request, the values already extracted for the previous parameters, and an instruction telling
   the model to *copy* the value rather than compute it.
2. A **checker** is chosen from the parameter type (`create_check`):

   | Parameter                                   | Checker         | Constraint                                                                                      |
   | ------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------- |
   | `number`                                    | `NumberCheck`   | Optional `-`, digits, at most one `.`. Ends on a terminator (space, newline, `,`, `)`).         |
   | `boolean`                                   | `PrefixCheck`   | Only `true` or `false` are allowed.                                                             |
   | name contains `regex` or `replace`          | `FreeTextCheck` | Any character. Ends on a closing quote or newline.                                              |
   | any other (strings)                         | `StringCheck`   | The generated text must remain a **substring of the user prompt** (case-insensitive).           |

3. The token loop runs (`get_next_valid_token`) until the checker declares the value complete, or
   until a safety limit of 32 steps is reached.
4. The raw text is converted to the target type (`parse_raw_value`): `int` or `float` for numbers,
   `bool` for booleans, a cleaned `str` otherwise.

### Step 3: selecting the next valid token (`get_next_valid_token`)

This is the core of the project.

```
logits ← LLM(input_ids)
best ← none
for each (token_id, token_text) in vocabulary:
    text ← decode byte-level markers ("Ġ" → " ", "Ċ" → "\n", "ĉ" → "\t")
    sim  ← deep copy of the checker
    for each character c of text:
        if not sim.is_valid(c): reject the token
        sim.add_to_generated(c)
    if the token was not rejected and logits[token_id] > best score:
        best ← token_id
return best            (RuntimeError if no token is valid)
```

Two details matter:

- **Tokens are not characters.** A token can contain several characters (`" Hello"`, `"123"`),
  while the state machines validate one character at a time. Each candidate token is therefore
  replayed character by character on a *copy* of the checker, so that the real state is not
  modified by tokens that are eventually rejected.
- **Termination is part of the grammar.** Delimiters (closing quote, newline, space after a
  number) are valid tokens like any other. Once the model chooses one, the checker switches to its
  "done" state and the loop ends. The delimiter itself is not part of the value.

### The state machines (`state_machines.py`)

Every checker exposes the same small interface, which is what allows the generation loop to be
generic:

| Method                | Role                                                          |
| --------------------- | ------------------------------------------------------------- |
| `is_valid(char)`      | Can this character come next, given what was generated so far? |
| `add_to_generated(c)` | Update the state (or switch to "done" on a delimiter).         |
| `is_complete()`       | Is the value finished?                                         |

## Design decisions

- **The LLM never writes JSON syntax.** The model only produces *values* (a function name, a
  number, a string). Braces, keys, quotes and commas are produced by `json.dump`. This removes a
  whole class of failures at no cost.
- **Two stages: name first, then parameters.** Once the function is known, the parameter names
  and types come from its definition, so each value can be constrained precisely and prompted with
  a targeted instruction. It also keeps each prompt short, which matters for a 0.6B model.
- **Character-level state machines applied to tokens.** Writing the constraints per character is
  much simpler and less error-prone than reasoning on the vocabulary directly, and the same
  interface works for every type. The price is the token-level simulation described above.
- **Extractive strings.** A string argument must be a substring of the user prompt. In function
  calling, string arguments are almost always copied from the request (`"Greet john"` gives
  `john`). This forbids hallucinated content and makes the output verifiable.
- **Greedy decoding.** The best-scoring valid token is always chosen. The program is therefore
  deterministic: the same input produces the same output.
- **Type-specific few-shot prompts.** Small models follow examples better than instructions. The
  numeric prompt explicitly says *not to compute* (a request like "square root of 144" must
  produce `144`, not `12`), and the string prompts show that verbs such as "Greet" or "Reverse"
  are not part of the argument.
- **Pydantic everywhere.** Input files are validated with Pydantic models (`FunctionDef`,
  `TypeSpec`, `PromptEntry`), and the checkers are Pydantic models too, which gives a cheap and
  safe `model_copy(deep=True)` for the token simulation.
- **Errors are reported, not thrown.** The loaders catch file and validation errors, print a
  readable message and return `None`. The entry point then stops cleanly.
- **Safety limit on generation.** Each parameter is capped at 32 steps so that a badly behaved
  generation can never loop forever.

## Performance analysis

### Reliability

- **Syntax:** the output is valid JSON in 100% of cases, since the structure is written by code.
- **Function names:** always one of the defined functions, thanks to `PrefixCheck`.
- **Types:** numbers are always parsed as numbers and booleans as booleans, whatever the model
  would have generated without constraints.
- **String arguments:** always a substring of the request, so they can never contain invented text.
- **Determinism:** greedy decoding gives identical results from one run to another.

What constrained decoding does **not** guarantee is *semantic* correctness: the model can still
pick the wrong function, or copy the wrong number. Constraints reduce the space of possible
mistakes, and prompt engineering reduces the remaining ones.

### Accuracy

Results on the provided test set:

| Metric                              | Result            |
| ----------------------------------- | ----------------- |
| Prompts processed                   | `11`             |
| Correct function selected           | `11 / 11`       |
| Fully correct parameters            | `8 / 11`       |
| Syntactically valid output          | 100% (by design)  |

The main semantic risks are the model being tempted to *solve* the request instead of extracting
from it, and ambiguity about which number is `a` and which is `b`. Both are addressed by the
few-shot prompts and the "first/second" ordinal wording.

### Speed

| Metric                       | Result   |
| ---------------------------- | -------- |
| Total time for the test set  | `3 min 50 sec`  |
| Average time per prompt      | `20 sec`  |

Each generated token costs one forward pass of the model plus a Python loop over the **entire
vocabulary** (about 150,000 tokens), with a deep copy of the checker for each candidate. This loop
dominates the running time. The current implementation favours clarity and correctness, and
several straightforward optimizations are known:

1. **Sort by logit, stop at the first valid token.** Iterating candidates from the highest to the
   lowest score and returning the first valid one gives exactly the same result while usually
   checking a handful of tokens instead of the whole vocabulary.
2. **Pre-compute the decoded text of every token** once at start-up instead of at every step.
3. **Avoid `model_copy(deep=True)`** for the simulation (for example by making the checkers
   expose a pure "would this text be valid?" method).
4. **Reuse the KV cache** of the model between steps instead of recomputing the whole sequence.

### Memory

The model is loaded once. The inverted vocabulary (`id → token`) is a single dictionary held for
the whole run. Nothing else grows with the number of prompts, except the list of results.

### Known limitations

- The `integer` type has no dedicated checker or parser yet: it goes through the string path
  instead of the numeric one.
- Only the byte-level markers for space, newline and tab are decoded. Tokens representing
  non-ASCII characters keep their raw byte-level form, so accented or exotic characters in string
  arguments are not handled reliably.
- A quote character (`'` or `"`) closes a string value, so an argument containing an apostrophe
  (for example `don't`) is cut at that point.
- `regex` and `replacement` parameters use `FreeTextCheck`, so they are syntactically constrained
  (single line, no quote) but not semantically validated as regular expressions.
- If no token of the vocabulary satisfies a constraint, a `RuntimeError` is raised and stops the
  program.

## Challenges faced

- **Tokens versus characters.** The constraints are naturally expressed on characters, but the
  model works on tokens that can hold several characters. *Solution:* simulate each candidate token
  on a copy of the checker, character by character, and accept it only if every character passes.
- **Byte-level BPE vocabulary.** The vocabulary file does not store tokens as plain text: a space
  appears as `Ġ` and a newline as `Ċ`. Without decoding them, no token containing a space would
  ever be valid. *Solution:* decode the markers before validating a token and before appending it
  to the checker.
- **Knowing when to stop.** A value has no natural end token. *Solution:* make delimiters part of
  the constraint (terminator after a number, closing quote or newline after a string) and switch
  the checker to a "done" state, plus a maximum number of steps as a safeguard.
- **The model computes instead of copying.** Asked for the number in "square root of 144", a
  small model happily answers `12`. *Solution:* few-shot examples showing the raw copy, and an
  explicit instruction not to solve anything.
- **Verbs leaking into arguments.** The model tends to include words like "Greet" or "Reverse" in
  the string value. *Solution:* dedicated examples and instruction, backed by the substring
  constraint.
- **Distinguishing several numbers.** For functions with two numeric parameters, a small
  model can easily pick the same number twice. *Solution:* give the ordinal ("first" / "second") in the
  prompt and include the values already extracted.
- **Different output shapes of the SDK.** Logits can come back nested in an extra list depending on
  the call. *Solution:* normalise them before use.
- **Speed.** A naive pass over the full vocabulary with deep copies is slow (see the performance
  section for the planned optimizations).

## Testing strategy

Validation relies on several complementary layers:

1. **Static checks.** `make lint` runs `flake8` and `mypy` with `--disallow-untyped-defs`,
   `--check-untyped-defs`, `--warn-return-any` and `--warn-unused-ignores`, so every function is
   fully typed and type errors are caught before running.
2. **Input validation.** Pydantic rejects malformed function definitions and prompts (wrong types,
   missing fields, unknown parameter types). Missing files, permission errors and invalid JSON
   are reported with an explicit message.
3. **End-to-end runs on the provided test set.** The program is run on
   `function_calling_tests.json` and the output is compared, prompt by prompt, with the expected
   function and arguments. Since decoding is greedy and deterministic, a failure can be reproduced
   exactly, then investigated with `make debug` (`pdb`).
4. **Isolated checkers.** The state machines are small, deterministic classes with no dependency on
   the LLM. They can be exercised directly with hand-written character sequences to verify that
   valid inputs are accepted and invalid ones rejected (for example `NumberCheck` must refuse `1.2.3`
   and `--5`, and `PrefixCheck` must refuse anything that is not a prefix of an allowed name).
5. **Edge cases to keep in mind when adding prompts:** negative numbers, decimals, several numbers
   in one prompt, strings with spaces or quotes, empty or very short prompts, and functions with no
   parameters.

## Project structure

```
.
├── Makefile
├── README.md
├── pyproject.toml
├── llm_sdk/                    # SDK provided with the subject (Small_LLM_Model)
├── data/
│   ├── input/                  # functions_definition.json, function_calling_tests.json
│   └── output/                 # generated results
└── src/
    ├── __main__.py             # entry point: orchestrates the whole pipeline
    ├── parser.py               # command-line arguments
    ├── parser_classes.py       # Pydantic models: TypeSpec, FunctionDef, PromptEntry
    ├── json_parsing.py         # loading and validation of the JSON files, vocabulary loading
    ├── prompt_engineering.py   # few-shot prompts, one per parameter kind
    ├── state_machines.py       # PrefixCheck, StringCheck, FreeTextCheck, NumberCheck
    └── generator.py            # constrained decoding loop and result writing
```

## Resources

### References

- [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), Willard
  and Louf, 2023: the paper behind the *Outlines* library, on constraining generation with
  finite-state machines.
- [XGrammar: Flexible and Efficient Structured Generation Engine for LLMs](https://arxiv.org/abs/2411.15100):
  a modern, optimized approach to grammar-constrained decoding.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388): description of the Qwen3 model
  family, including the 0.6B model used here.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909),
  Sennrich et al., 2016: the original BPE tokenization paper.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
  the GPT-2 paper, which introduced byte-level BPE (source of the `Ġ` / `Ċ` markers).
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers): tokenizers,
  logits and text generation.
- [Pydantic documentation](https://docs.pydantic.dev/): models, validation and `TypeAdapter`.
- [JSON Schema](https://json-schema.org/understanding-json-schema/): primitive types used to
  describe parameters.
- [uv documentation](https://docs.astral.sh/uv/): project and environment management.
- [Outlines](https://github.com/dottxt-ai/outlines) and
  [llama.cpp GBNF grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md):
  existing implementations of constrained decoding, used for comparison.

### Use of AI

AI (Claude, by Anthropic) was used for the following tasks:

- **README:** drafting and structuring this file from the source code and the requirements of the
  subject. The content was reviewed and checked against the code.
- Create some promts tests to look at the reaction of the llm.
