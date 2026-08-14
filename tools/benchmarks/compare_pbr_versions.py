"""Aggregate multi-scene, multi-round V2.2 Stylized versus V3 PBR results."""

import argparse
import json
import os

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


COLUMNS = ["V2.2 Original", "V2.2 Stylized Metal", "V3 Dielectric",
           "V3 Metal", "V3 Glass"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output_dir", default="comparisons/pbr_report")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    for path in args.metrics:
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))

    fig, axes = plt.subplots(len(records), len(COLUMNS),
                             figsize=(3.4 * len(COLUMNS), 3.0 * len(records)))
    if len(records) == 1:
        axes = axes[None]
    summary = {"datasets": {}, "all_finite": True}
    for row, record in enumerate(records):
        panel = iio.imread(record["panel"])
        pieces = np.array_split(panel, len(COLUMNS), axis=1)
        for column, (title, image) in enumerate(zip(COLUMNS, pieces)):
            axes[row, column].imshow(image)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(title)
        axes[row, 0].set_ylabel(
            f"{record['scene']}\n{record['metrics']['runtime']['gaussians']:,} Gs")
        metrics = record["metrics"]
        rounds = [metrics[name] for name in ("dielectric", "metal", "glass")]
        v3_time = [metrics["runtime"]["primary_seconds"]
                   + metrics["runtime"]["shadow_seconds"] + item["secondary_seconds"]
                   for item in rounds]
        finite = min(item["finite_fraction"] for item in rounds)
        summary["all_finite"] = summary["all_finite"] and finite == 1.0
        summary["datasets"][record["scene"]] = {
            "gaussians": metrics["runtime"]["gaussians"],
            "resolution": [metrics["runtime"]["width"], metrics["runtime"]["height"]],
            "v22_stylized_mean_luminance": metrics["v22_stylized"]["mean_luminance"],
            "v3_mean_luminance": {name: metrics[name]["mean_luminance"]
                                  for name in ("dielectric", "metal", "glass")},
            "v3_mean_full_frame_seconds": float(np.mean(v3_time)),
            "v3_estimated_fps": float(1.0 / np.mean(v3_time)),
            "shadow_mean_visibility": metrics["runtime"]["mean_visibility"],
            "minimum_finite_fraction": finite,
        }
    fig.tight_layout()
    figure_path = os.path.join(args.output_dir, "v22_vs_v3_multi_scene.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    summary["figure"] = os.path.relpath(figure_path)
    summary["method"] = (
        "Same trained Gaussian model and camera. V2.2 control uses its Metal SH Edit; "
        "V3 rounds use manual PBR parameters, HDR, GGX, shadow visibility, reflection and "
        "refraction rays. CUDA timings cover synchronized primary, shadow and secondary tracing; "
        "they exclude property-map rasterization, GUI work and image encoding."
    )
    with open(os.path.join(args.output_dir, "aggregate_metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
