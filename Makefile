FUNC_DEF ?= data/input/functions_definition.json
INPUT    ?= data/input/function_calling_tests.json
OUTPUT   ?= data/output/function_calls.json

install:
	uv sync

run:
	HF_HOME=/home/mgerard/sgoinfre/students/mgerard/.cache/huggingface uv run python -m src \
		--functions_definition $(FUNC_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	HF_HOME=/home/mgerard/sgoinfre/students/mgerard/.cache/huggingface uv run python -m pdb -m src \
		--functions_definition $(FUNC_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	-uv run flake8 . --exclude=.venv,llm_sdk
	uv run mypy . --exclude "(.venv|sdk)" --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict
