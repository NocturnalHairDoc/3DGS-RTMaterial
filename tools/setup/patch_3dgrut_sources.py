#!/usr/bin/env python3
"""Make upstream 3DGRT installable in this project's Python 3.10 environment."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "3dgrut"
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        source = pyproject.read_text()
        patched = source.replace('requires-python = ">=3.11"', 'requires-python = ">=3.10"')
        if patched != source:
            pyproject.write_text(patched)
            print(f"Patched Python requirement: {pyproject}")

    timer = root / "threedgrut" / "utils" / "timer.py"
    if timer.is_file():
        source = timer.read_text()
        old = "from typing import Callable, Final, Optional, Self, Type, TypeVar, cast"
        if old in source:
            replacement = (
                "from typing import Callable, Final, Optional, Type, TypeVar, cast\n"
                "try:\n"
                "    from typing import Self\n"
                "except ImportError:\n"
                "    from typing_extensions import Self"
            )
            timer.write_text(source.replace(old, replacement))
            print(f"Patched typing.Self compatibility: {timer}")

    package_init = root / "threedgrut" / "__init__.py"
    if package_init.is_file():
        source = package_init.read_text()
        old = "import tomllib"
        if "except ModuleNotFoundError:\n    import tomli as tomllib" not in source and old in source:
            replacement = (
                "try:\n"
                "    import tomllib\n"
                "except ModuleNotFoundError:\n"
                "    import tomli as tomllib"
            )
            package_init.write_text(source.replace(old, replacement, 1))
            print(f"Patched tomllib compatibility: {package_init}")


if __name__ == "__main__":
    main()
