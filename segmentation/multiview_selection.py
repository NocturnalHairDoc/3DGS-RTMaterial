"""Interactive selection manifests for cross-view SAM-to-Gaussian fusion.

The manifest records only camera/image names and SAM proposal indices.  Camera
poses continue to come from the scene's COLMAP reconstruction, so selections
made in several photographs can be projected into one Gaussian coordinate
system by :mod:`segmentation.sam_driven`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


MANIFEST_VERSION = 1


def proposal_indices_by_area(masks: torch.Tensor, minimum=0.001, maximum=0.20):
    """Return proposal indices whose image-area fractions are in range."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError("SAM masks must have shape (M, H, W)")
    fractions = masks.flatten(1).float().mean(1)
    keep = (fractions >= float(minimum)) & (fractions <= float(maximum))
    return torch.nonzero(keep, as_tuple=False).flatten().tolist()


def smallest_proposal_at(masks: torch.Tensor, x: int, y: int, candidates=None):
    """Choose the smallest candidate SAM proposal containing pixel ``(x, y)``."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    height, width = masks.shape[-2:]
    x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
    indices = (list(range(masks.shape[0])) if candidates is None
               else [int(index) for index in candidates])
    containing = [index for index in indices if bool(masks[index, y, x])]
    if not containing:
        return None
    areas = masks[containing].flatten(1).sum(1)
    return int(containing[int(torch.argmin(areas))])


def largest_proposal_at(masks: torch.Tensor, x: int, y: int, candidates=None):
    """Choose the largest eligible proposal containing pixel ``(x, y)``."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    height, width = masks.shape[-2:]
    x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
    indices = (list(range(masks.shape[0])) if candidates is None
               else [int(index) for index in candidates])
    containing = [index for index in indices if bool(masks[index, y, x])]
    if not containing:
        return None
    areas = masks[containing].flatten(1).sum(1)
    return int(containing[int(torch.argmax(areas))])


def proposals_along_stroke(masks: torch.Tensor, points, candidates=None,
                           prefer_largest=False):
    """Collect eligible proposals touched at each stroke sample."""
    chooser = largest_proposal_at if prefer_largest else smallest_proposal_at
    selected = set()
    for x, y in points:
        index = chooser(masks, x, y, candidates)
        if index is not None:
            selected.add(int(index))
    return selected


def smallest_proposals_along_stroke(masks: torch.Tensor, points, candidates=None):
    """Backward-compatible fine-parts stroke selection."""
    return proposals_along_stroke(
        masks, points, candidates=candidates, prefer_largest=False)


def save_selection_manifest(path, source_path, masks_dir, selections, **settings):
    payload = {
        "version": MANIFEST_VERSION,
        "source_path": str(Path(source_path).expanduser().resolve()),
        "masks_dir": str(Path(masks_dir).expanduser().resolve()),
        "views": [
            {"image": str(name), "mask_indices": sorted({int(v) for v in indices})}
            for name, indices in sorted(selections.items()) if indices
        ],
    }
    payload.update(settings)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_selection_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"unsupported multi-view manifest version: {payload.get('version')}")
    views = payload.get("views")
    if not isinstance(views, list) or len(views) < 2:
        raise ValueError("multi-view selection needs at least two selected photographs")
    selections = {}
    for view in views:
        stem = Path(str(view["image"])).stem
        indices = sorted({int(value) for value in view.get("mask_indices", [])})
        if indices:
            selections[stem] = indices
    if len(selections) < 2:
        raise ValueError("multi-view selection needs masks in at least two photographs")
    source_path = Path(payload["source_path"]).expanduser().resolve()
    masks_dir = Path(payload["masks_dir"]).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"selection source path does not exist: {source_path}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"selection mask directory does not exist: {masks_dir}")
    return payload, source_path, masks_dir, selections


def launch_selector(source_path, masks_dir, output_path, minimum=0.001, maximum=0.20):
    """Open a small native photo/mask selector and save a manifest.

    Left click or drag paints the object by accumulating the smallest eligible
    SAM proposal below each stroke sample. Right click or drag erases touched
    proposals. All selected proposals in one photograph are unioned before 3D
    fusion, so that photograph still contributes at most one cross-view vote.
    """
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk

    source_path, masks_dir = Path(source_path), Path(masks_dir)
    image_dir = source_path / "images"
    mask_paths = sorted(masks_dir.glob("*.pt"))
    if len(mask_paths) < 2:
        raise ValueError(f"need masks for at least two views under {masks_dir}")

    root = tk.Tk()
    root.title("3DGS single-object multi-view selection")
    canvas = tk.Canvas(root, width=1100, height=740, background="black")
    canvas.grid(row=0, column=0, columnspan=8, sticky="nsew")
    status = tk.StringVar()
    selection_policy = tk.StringVar(value="whole")
    tk.Label(root, textvariable=status, anchor="w").grid(
        row=2, column=0, columnspan=8, sticky="ew")
    selections = {path.stem: set() for path in mask_paths}
    state = {"index": 0, "masks": None, "scale": 1.0, "photo": None,
             "candidates": None, "last_xy": None}

    def image_path(stem):
        matches = list(image_dir.glob(stem + ".*"))
        if not matches:
            raise FileNotFoundError(f"training image not found for {stem}")
        return matches[0]

    def redraw():
        path = mask_paths[state["index"]]
        masks = torch.load(path, map_location="cpu", weights_only=False).bool()
        photo = Image.open(image_path(path.stem)).convert("RGB").resize(
            (masks.shape[-1], masks.shape[-2]))
        rgb = np.asarray(photo).copy()
        selected = sorted(selections[path.stem])
        if selected:
            union = torch.any(masks[selected], dim=0).numpy()
            color = np.asarray([40, 230, 90], dtype=np.uint8)
            rgb[union] = (0.42 * rgb[union] + 0.58 * color).astype(np.uint8)
        display = Image.fromarray(rgb)
        scale = min(1100 / display.width, 740 / display.height)
        display = display.resize((round(display.width * scale), round(display.height * scale)))
        state.update(
            masks=masks, scale=scale, photo=ImageTk.PhotoImage(display),
            candidates=proposal_indices_by_area(masks, minimum, maximum))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        status.set(
            f"View {state['index'] + 1}/{len(mask_paths)}: {path.stem} | "
            f"selected proposals: {selected if selected else 'none'} | "
            f"mode: {'whole object' if selection_policy.get() == 'whole' else 'fine parts'} | "
            "left drag: paint/add, right drag: erase")

    def paint(event, remove=False, begin=False):
        path = mask_paths[state["index"]]
        current = np.asarray([float(event.x), float(event.y)], dtype=np.float32)
        previous = current if begin or state["last_xy"] is None else state["last_xy"]
        distance = float(np.linalg.norm(current - previous))
        sample_count = max(1, int(np.ceil(distance / 4.0)) + 1)
        display_points = np.linspace(previous, current, sample_count)
        points = [(x / state["scale"], y / state["scale"])
                  for x, y in display_points]
        candidate_pool = (sorted(selections[path.stem])
                          if remove else state["candidates"])
        indices = proposals_along_stroke(
            state["masks"], points, candidate_pool,
            prefer_largest=(not remove and selection_policy.get() == "whole"))
        before = set(selections[path.stem])
        if remove:
            selections[path.stem].difference_update(indices)
        else:
            selections[path.stem].update(indices)
        state["last_xy"] = current
        if before != selections[path.stem]:
            redraw()

    def end_stroke(_event):
        state["last_xy"] = None

    def move(delta):
        state["index"] = (state["index"] + delta) % len(mask_paths)
        redraw()

    def save():
        active = {name: values for name, values in selections.items() if values}
        if len(active) < 2:
            messagebox.showerror("Selection incomplete", "Select masks in at least two photographs.")
            return
        if len(active) > 12 and not messagebox.askyesno(
                "Many selected views",
                f"{len(active)} views are selected. 3-8 diverse, clean views are usually "
                "safer and faster. Save all selected views anyway?"):
            return
        output = save_selection_manifest(
            output_path, source_path, masks_dir, active,
            min_votes=2, min_mask_area_fraction=minimum,
            max_mask_area_fraction=maximum,
            minimum_visible_support_fraction=0.50,
            max_instance_fraction=0.50,
            selection_mode="single_object",
        )
        messagebox.showinfo("Saved", str(output))

    canvas.bind("<ButtonPress-1>", lambda event: paint(event, False, True))
    canvas.bind("<B1-Motion>", lambda event: paint(event, False))
    canvas.bind("<ButtonRelease-1>", end_stroke)
    canvas.bind("<ButtonPress-3>", lambda event: paint(event, True, True))
    canvas.bind("<B3-Motion>", lambda event: paint(event, True))
    canvas.bind("<ButtonRelease-3>", end_stroke)
    tk.Button(root, text="Previous photo", command=lambda: move(-1)).grid(row=1, column=0)
    tk.Button(root, text="Next photo", command=lambda: move(1)).grid(row=1, column=1)
    tk.Button(root, text="Clear this photo", command=lambda: (
        selections[mask_paths[state["index"]].stem].clear(), redraw())).grid(row=1, column=2)
    tk.Button(root, text="Save selection", command=save).grid(row=1, column=3)
    tk.Button(root, text="Close", command=root.destroy).grid(row=1, column=4)
    tk.Radiobutton(root, text="Whole-object brush", variable=selection_policy,
                   value="whole", command=redraw).grid(row=1, column=5)
    tk.Radiobutton(root, text="Fine-parts brush", variable=selection_policy,
                   value="parts", command=redraw).grid(row=1, column=6)
    redraw()
    root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select one object in calibrated photos")
    parser.add_argument("--source", required=True, help="COLMAP scene containing images/ and sparse/")
    parser.add_argument("--masks", required=True, help="SAM .pt mask directory")
    parser.add_argument("--output", required=True, help="selection JSON to create")
    parser.add_argument("--min-area", type=float, default=0.001)
    parser.add_argument("--max-area", type=float, default=0.50)
    args = parser.parse_args()
    launch_selector(args.source, args.masks, args.output, args.min_area, args.max_area)


if __name__ == "__main__":
    main()
