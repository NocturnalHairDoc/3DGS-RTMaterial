"""Generate SAGA-compatible per-image masks with Meta SAM 2.1."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from windows_bootstrap import configure_windows_runtime


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract SAGA masks with SAM 2.1")
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints" / "sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--downsample", choices=("1", "2", "4", "8"), default="4")
    parser.add_argument("--mask_size", type=int, default=200)
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only N images; useful for validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_windows_runtime()

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2 mask extraction requires a CUDA GPU.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 2 checkpoint not found: {args.checkpoint}\n"
            "Run .\\install_sam2.ps1 first or pass --checkpoint."
        )

    image_dir = args.image_root / ("images" if args.downsample == "1" else f"images_{args.downsample}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    output_dir = args.image_root / "sam_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No supported images found in {image_dir}")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    print(f"Loading SAM 2.1: {args.checkpoint}")
    model = build_sam2(args.model_cfg, str(args.checkpoint), device="cuda", apply_postprocessing=True)
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    written = skipped = 0
    with torch.inference_mode(), autocast:
        for image_path in tqdm(images, desc="SAM 2 masks"):
            output_path = output_dir / f"{image_path.stem}.pt"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue

            # A writable copy avoids TorchVision's warning about PIL-backed,
            # read-only NumPy views on Windows.
            image = np.array(Image.open(image_path).convert("RGB"), copy=True)
            records = generator.generate(image)
            if not records:
                raise RuntimeError(f"SAM 2 returned no masks for {image_path}")

            masks = np.stack([record["segmentation"] for record in records], axis=0)
            masks_tensor = torch.from_numpy(masks).to(dtype=torch.float32).unsqueeze(1)
            masks_tensor = F.interpolate(
                masks_tensor,
                size=(args.mask_size, args.mask_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            masks_tensor = (masks_tensor >= 0.5).to(dtype=torch.float32).cpu()
            torch.save(masks_tensor, output_path)
            written += 1

    print(f"SAM 2 extraction complete: written={written}, skipped={skipped}, output={output_dir}")


if __name__ == "__main__":
    main()
