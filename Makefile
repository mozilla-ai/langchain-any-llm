.PHONY: all format lint lint_package lint_tests test tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

# Run unit tests
test tests:
	uv run --group test pytest tests/

# Run integration tests
integration_tests:
	uv run --group test pytest tests/integration_tests/

######################
# LINTING AND FORMATTING
######################

# Format code using ruff
format format_diff:
	uv run --group lint ruff format .
	uv run --group lint ruff check . --fix

# Lint code
lint lint_diff:
	uv run --group lint ruff check .
	uv run --group lint mypy .

# Lint package code only
lint_package:
	uv run --group lint ruff check langchain_anyllm/
	uv run --group lint mypy langchain_anyllm/

# Lint test code only
lint_tests:
	uv run --group lint ruff check tests/
	uv run --group lint mypy tests/

# Spell check
spell_check:
	uv run --group codespell codespell --toml pyproject.toml

# Fix spelling
spell_fix:
	uv run --group codespell codespell --toml pyproject.toml -w

######################
# PRE-COMMIT
######################

# Install pre-commit hooks
pc-install:
	uv run --group dev pre-commit install

# Run pre-commit on all files
pc-run:
	uv run --group dev pre-commit run --all-files

# Update pre-commit hooks
pc-update:
	uv run --group dev pre-commit autoupdate

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters on all code'
	@echo 'lint_package                 - run linters on package code only'
	@echo 'lint_tests                   - run linters on test code only'
	@echo 'test                         - run unit tests'
	@echo 'integration_tests            - run integration tests'
	@echo 'spell_check                  - run codespell on the project'
	@echo 'spell_fix                    - run codespell on the project and fix the errors'
	@echo 'pc-install                   - install pre-commit hooks'
	@echo 'pc-run                       - run pre-commit on all files'
	@echo 'pc-update                    - update pre-commit hooks'
