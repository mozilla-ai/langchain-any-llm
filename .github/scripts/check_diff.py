import json
import sys
from typing import Dict

# Root directory since we flattened the structure
LIB_DIRS = ["."]

if __name__ == "__main__":
    files = sys.argv[1:]

    dirs_to_run: Dict[str, set] = {
        "lint": set(),
        "test": set(),
    }

    if len(files) == 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed files.")

    for file in files:
        # Skip non-code files
        if file.startswith(".git"):
            continue

        # For any Python file, test file, or pyproject.toml change, run tests
        if file.endswith((".py", ".toml")) or file.startswith(("langchain_anyllm/", "tests/", "examples/")):
            dirs_to_run["test"].add(".")

        # For workflow changes, run tests
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            dirs_to_run["test"].update(LIB_DIRS)

    outputs = {
        "dirs-to-lint": list(dirs_to_run["lint"] | dirs_to_run["test"]),
        "dirs-to-test": list(dirs_to_run["test"]),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
