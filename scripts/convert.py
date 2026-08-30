#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

"""Convert raw photographs into the COLMAP layout used by 3DGS training.

This module is based on the Mip-NeRF 360 conversion script used by the original
3D Gaussian Splatting project. External commands are passed as argument lists,
so dataset and executable paths may safely contain spaces.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Convert raw images to a COLMAP dataset")
    parser.add_argument("--no_gpu", action="store_true", help="disable COLMAP GPU matching")
    parser.add_argument(
        "--skip_matching",
        action="store_true",
        help="reuse an existing distorted/database.db and distorted/sparse/0",
    )
    parser.add_argument("--source_path", "-s", required=True, type=Path)
    parser.add_argument("--camera", default="OPENCV")
    parser.add_argument(
        "--colmap_executable",
        default="colmap",
        help="COLMAP executable path (default: colmap from PATH)",
    )
    parser.add_argument("--resize", action="store_true")
    parser.add_argument(
        "--magick_executable",
        default="",
        help="ImageMagick 'magick' executable; standalone mogrify is used when omitted",
    )
    return parser


def build_colmap_commands(args: Namespace, source: Path) -> list[list[str]]:
    """Build the COLMAP commands without executing them."""
    colmap = str(args.colmap_executable or "colmap")
    input_dir = source / "input"
    distorted = source / "distorted"
    database = distorted / "database.db"
    sparse = distorted / "sparse"
    use_gpu = "0" if args.no_gpu else "1"
    commands: list[list[str]] = []

    if not args.skip_matching:
        commands.extend(
            [
                [
                    colmap,
                    "feature_extractor",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(input_dir),
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.camera_model",
                    str(args.camera),
                    "--SiftExtraction.use_gpu",
                    use_gpu,
                ],
                [
                    colmap,
                    "exhaustive_matcher",
                    "--database_path",
                    str(database),
                    "--SiftMatching.use_gpu",
                    use_gpu,
                ],
                [
                    colmap,
                    "mapper",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(input_dir),
                    "--output_path",
                    str(sparse),
                    "--Mapper.ba_global_function_tolerance=0.000001",
                ],
            ]
        )

    commands.append(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(input_dir),
            "--input_path",
            str(sparse / "0"),
            "--output_path",
            str(source),
            "--output_type",
            "COLMAP",
        ]
    )
    return commands


def build_resize_command(args: Namespace, image: Path, scale: str) -> list[str]:
    if args.magick_executable:
        return [str(args.magick_executable), "mogrify", "-resize", scale, str(image)]
    return ["mogrify", "-resize", scale, str(image)]


def run_command(command: Sequence[str]) -> None:
    logging.info("Running: %s", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), check=True)


def _validate_inputs(args: Namespace, source: Path) -> None:
    input_dir = source / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input image directory does not exist: {input_dir}")
    if args.skip_matching:
        sparse_model = source / "distorted" / "sparse" / "0"
        if not sparse_model.is_dir():
            raise FileNotFoundError(
                "--skip_matching requires an existing sparse model at "
                f"{sparse_model}"
            )


def _normalize_sparse_layout(source: Path) -> None:
    sparse_root = source / "sparse"
    if not sparse_root.is_dir():
        raise FileNotFoundError(f"COLMAP did not create the sparse directory: {sparse_root}")
    sparse_zero = sparse_root / "0"
    sparse_zero.mkdir(exist_ok=True)
    for item in list(sparse_root.iterdir()):
        if item == sparse_zero:
            continue
        destination = sparse_zero / item.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing COLMAP output: {destination}")
        shutil.move(str(item), str(destination))


def _create_resized_images(args: Namespace, source: Path) -> None:
    image_root = source / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"undistorted image directory does not exist: {image_root}")
    print("Copying and resizing images...")
    variants = (("images_2", "50%"), ("images_4", "25%"), ("images_8", "12.5%"))
    for directory, _ in variants:
        (source / directory).mkdir(exist_ok=True)
    for image in sorted(path for path in image_root.iterdir() if path.is_file()):
        for directory, scale in variants:
            destination = source / directory / image.name
            shutil.copy2(image, destination)
            run_command(build_resize_command(args, destination, scale))


def convert_dataset(args: Namespace) -> Path:
    source = args.source_path.expanduser().resolve()
    _validate_inputs(args, source)
    if not args.skip_matching:
        (source / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)
    for command in build_colmap_commands(args, source):
        run_command(command)
    _normalize_sparse_layout(source)
    if args.resize:
        _create_resized_images(args, source)
    return source


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        source = convert_dataset(args)
    except subprocess.CalledProcessError as exc:
        logging.error("External command failed with exit code %s", exc.returncode)
        return int(exc.returncode or 1)
    except (FileNotFoundError, FileExistsError, OSError) as exc:
        logging.error("Conversion failed: %s", exc)
        return 1
    print(f"Done: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
