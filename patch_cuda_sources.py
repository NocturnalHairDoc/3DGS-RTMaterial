#!/usr/bin/env python3
"""Apply small compatibility fixes to freshly downloaded SAGA extensions."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "submodules"
    headers = sorted(root.glob("diff-gaussian-rasterization*/cuda_rasterizer/rasterizer_impl.h"))
    if not headers:
        raise SystemExit(f"No rasterizer headers found under {root}")
    for header in headers:
        source = header.read_text()
        if "#include <cstdint>" not in source:
            source = source.replace("#include <iostream>", "#include <cstdint>\n#include <iostream>")
            header.write_text(source)
            print(f"Patched CUDA 12 compatibility: {header}")

    simple_knn = root / "simple-knn" / "simple_knn.cu"
    if not simple_knn.is_file():
        raise SystemExit(f"Missing source: {simple_knn}")
    source = simple_knn.read_text()
    if "#include <cfloat>" not in source:
        first_include = source.find("#include")
        source = source[:first_include] + "#include <cfloat>\n" + source[first_include:]
        simple_knn.write_text(source)
        print(f"Patched CUDA 12 compatibility: {simple_knn}")


if __name__ == "__main__":
    main()
