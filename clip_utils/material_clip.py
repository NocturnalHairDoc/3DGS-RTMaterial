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


def blend_material_params_from_scores(scores, temperature=0.10):
    material_names = list(MATERIAL_PARAM_PRESETS.keys())
    logits = torch.tensor([scores[m] for m in material_names], dtype=torch.float32)
    weights = torch.softmax(logits / max(temperature, 1e-6), dim=0)

    specular_gain = 0.0
    saturation = 0.0
    opacity_scale = 0.0
    tint = torch.zeros(3, dtype=torch.float32)

    for idx, material_name in enumerate(material_names):
        w = float(weights[idx].item())
        preset = MATERIAL_PARAM_PRESETS[material_name]
        specular_gain += w * preset["specular_gain"]
        saturation += w * preset["saturation"]
        opacity_scale += w * preset["opacity_scale"]
        tint += w * torch.tensor(preset["tint"], dtype=torch.float32)

    return {
        "strength": 1.0,
        "specular_gain": float(specular_gain),
        "saturation": float(saturation),
        "opacity_scale": float(opacity_scale),
        "tint": [float(x) for x in tint.tolist()],
    }

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
