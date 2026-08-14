import torch

COLOR_TINTS = {
    "red": [1.10, 0.85, 0.85],
    "blue": [0.85, 0.95, 1.10],
    "green": [0.85, 1.05, 0.85],
    "gold": [1.08, 0.96, 0.72],
    "silver": [1.00, 1.00, 1.02],
    "black": [0.70, 0.70, 0.70],
    "white": [1.05, 1.05, 1.05],
}

KEYWORD_PARAM_RULES = {
    "metal": {"specular_gain": 1.6, "saturation": 0.6, "opacity_scale": 1.0},
    "glass": {"specular_gain": 1.1, "saturation": 0.8, "opacity_scale": 0.25},
    "plastic": {"specular_gain": 0.2, "saturation": 1.2, "opacity_scale": 1.0},
    "matte": {"specular_gain": 0.0, "saturation": 1.0, "opacity_scale": 1.0},
    "brushed": {"specular_gain": 1.2},
    "polished": {"specular_gain": 1.8},
    "frosted": {"opacity_scale": 0.45, "saturation": 0.9},
    "transparent": {"opacity_scale": 0.2},
    "opaque": {"opacity_scale": 1.0},
    "shiny": {"specular_gain": 1.7},
    "rough": {"specular_gain": 0.4},
    "acrylic": {"specular_gain": 0.2, "saturation": 1.2, "opacity_scale": 0.7},
}

MATERIAL_CLIP_PROMPTS = {
    "Metal": [
        "a shiny metal object",
        "a chrome metal surface",
        "a brushed metal object",
    ],
    "Glass": [
        "a glass object",
        "a transparent glass surface",
        "a clear reflective glass object",
    ],
    "Plastic": [
        "a plastic object",
        "a smooth plastic surface",
        "a colored plastic object",
    ],
    "Matte": [
        "a matte surface",
        "a diffuse ceramic object",
        "a non reflective painted surface",
    ],
}

MATERIAL_PARAM_PRESETS = {
    "Metal": {
        "strength": 1.0,
        "specular_gain": 1.5,
        "saturation": 0.6,
        "opacity_scale": 1.0,
        "tint": [1.0, 0.98, 0.95],
    },
    "Glass": {
        "strength": 1.0,
        "specular_gain": 1.1,
        "saturation": 0.7,
        "opacity_scale": 0.3,
        "tint": [0.92, 0.97, 1.05],
    },
    "Plastic": {
        "strength": 1.0,
        "specular_gain": 0.2,
        "saturation": 1.2,
        "opacity_scale": 1.0,
        "tint": [1.0, 1.0, 1.0],
    },
    "Matte": {
        "strength": 1.0,
        "specular_gain": 0.0,
        "saturation": 1.0,
        "opacity_scale": 1.0,
        "tint": [1.0, 1.0, 1.0],
    },
}


@torch.no_grad()
def score_crop_against_material_prompts(clip_model, crop_bchw):
    crop_bchw = crop_bchw.to("cuda", non_blocking=True).float().clamp(0.0, 1.0)

    image_features = clip_model.encode_image(crop_bchw)
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)

    scores = {}
    for material_name, prompts in MATERIAL_CLIP_PROMPTS.items():
        tokens = torch.cat([clip_model.tokenizer(p) for p in prompts]).to("cuda")
        text_features = clip_model.model.encode_text(tokens)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        sims = image_features @ text_features.T
        scores[material_name] = float(sims.mean().item())

    return scores


def topk_materials(scores, k=3):
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


def aggregate_multiview_scores(view_scores):
    """Robust mean material scores from one or more independently rendered views."""
    if not view_scores:
        raise ValueError("at least one view score is required")
    names = set(view_scores[0])
    if any(set(scores) != names for scores in view_scores):
        raise ValueError("all views must score the same materials")
    return {name: float(sum(scores[name] for scores in view_scores) / len(view_scores))
            for name in sorted(names)}


def select_confident_material(scores, min_score=0.15, min_margin=0.02):
    ranked = topk_materials(scores, 2)
    if not ranked:
        return None, 0.0
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
    if ranked[0][1] < min_score or margin < min_margin:
        return None, float(margin)
    return ranked[0][0], float(margin)


def masked_original_rgb_crop(original_chw, mask_hw, padding=8, outside=0.5):
    """Crop an independently rendered original RGB using only its segment mask."""
    if original_chw.ndim != 3 or original_chw.shape[0] != 3 or mask_hw.ndim != 2:
        raise ValueError("expected RGB (3,H,W) and mask (H,W)")
    visible = mask_hw > 0.10
    if not visible.any():
        raise ValueError("selected segment is not visible")
    ys, xs = torch.where(visible)
    h, w = mask_hw.shape
    y0, y1 = max(0, int(ys.min()) - padding), min(h, int(ys.max()) + padding + 1)
    x0, x1 = max(0, int(xs.min()) - padding), min(w, int(xs.max()) + padding + 1)
    soft = mask_hw[y0:y1, x0:x1].clamp(0, 1).unsqueeze(0)
    rgb = original_chw[:, y0:y1, x0:x1].clamp(0, 1)
    return rgb * soft + float(outside) * (1.0 - soft)


def params_from_text_prompt(text_prompt: str) -> dict:
    text = text_prompt.lower()

    params = {
        "strength": 1.0,
        "specular_gain": 1.0,
        "saturation": 1.0,
        "opacity_scale": 1.0,
        "tint": [1.0, 1.0, 1.0],
    }

    for key, rule in KEYWORD_PARAM_RULES.items():
        if key in text:
            for k, v in rule.items():
                params[k] = v

    tint = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    for color_name, rgb in COLOR_TINTS.items():
        if color_name in text:
            tint = torch.tensor(rgb, dtype=torch.float32)
            break

    params["tint"] = [float(x) for x in tint.tolist()]
    return params
