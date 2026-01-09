.PHONY: all format lint test tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

# Run unit tests
test tests:
	uv run pytest tests/

# Run integration tests
integration_tests:
	uv run pytest tests/integration_tests/

######################
# LINTING AND FORMATTING
######################

# Format code using ruff
format format_diff:
	uv run ruff format .
	uv run ruff check . --fix

# Lint code
lint lint_diff:
	uv run ruff check .
	uv run mypy .

# Spell check
spell_check:
	uv run codespell --toml pyproject.toml

# Fix spelling
spell_fix:
	uv run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'integration_tests            - run integration tests'
	@echo 'spell_check                  - run codespell on the project'
	@echo 'spell_fix                    - run codespell on the project and fix the errors'
