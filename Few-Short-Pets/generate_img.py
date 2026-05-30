"""
Generate CLIP-training images for 37 pet classes using LoRA-finetuned SD 1.5.

Each image gets a UNIQUE prompt built from diverse combinations of:
  pose × location × lighting × style × quality_suffix

This maximises visual diversity for CLIP ViT-B/16 fine-tuning.

Usage:
    python generate_lora_images.py \
        --img_dir    ~/F/Few-Short-Pets/img \
        --lora_path  /path/to/lora.safetensors \
        --output_dir ~/F/Few-Short-Pets/generated \
        [--base_model runwayml/stable-diffusion-v1-5] \
        [--images_per_class 12] \
        [--lora_scale 0.8] \
        [--steps 30] \
        [--guidance 7.5] \
        [--width 512] [--height 512] \
        [--seed 42] \
        [--save_prompt_log]   # saves prompts_log.json next to output_dir
"""

import argparse
import json
import re
import torch
from itertools import product
from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# ══════════════════════════════════════════════════════════════════
#  Diversity pools  (CLIP training: maximise visual distribution)
# ══════════════════════════════════════════════════════════════════

POSES = [
    "sitting and looking at the camera",
    "lying down on its side",
    "standing and looking sideways",
    "running on grass",
    "playing with a toy",
    "yawning with mouth open",
    "curled up sleeping",
    "jumping mid-air",
    "sniffing the ground",
    "looking up",
    "stretching front paws forward",
    "walking toward the camera",
]

LOCATIONS = [
    "in a sunny garden",
    "on a wooden floor indoors",
    "in a park with green trees",
    "on a couch near a window",
    "on a dirt path in the forest",
    "on a white studio background",
    "on a sandy beach",
    "on a stone patio",
    "on a grassy lawn",
    "inside a cozy living room",
    "near a pond in nature",
    "on a concrete sidewalk",
]

LIGHTINGS = [
    "natural daylight",
    "soft indoor lighting",
    "golden hour sunlight",
    "overcast diffuse light",
    "dramatic side lighting",
    "bright midday sun",
]

STYLES = [
    "photorealistic candid photo",
    "professional pet photography",
    "amateur smartphone photo",
    "DSLR shallow depth of field photo",
]

QUALITY_SUFFIX = "highly detailed, sharp focus, 8k resolution"

NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, ugly, bad anatomy, extra limbs, "
    "mutation, watermark, text, cartoon, sketch, painting, anime, "
    "out of frame, cropped, worst quality, low resolution, "
    "human hands, person in background, collage, multiple animals"
)

# ──────────────────────────────────────────────
#  Per-class breed descriptor  (injected into every prompt)
#  Focuses on visually distinctive traits CLIP needs to learn
# ──────────────────────────────────────────────
BREED_DESCRIPTORS = {
    "Abyssinian":               "Abyssinian cat with ticked tabby coat and slender athletic body",
    "American_Bulldog":         "American Bulldog with muscular build and white brindle coat",
    "American_Pit_Bull_Terrier":"American Pit Bull Terrier with athletic build and short smooth coat",
    "Basset_Hound":             "Basset Hound with long droopy ears and tricolor coat",
    "Beagle":                   "Beagle with tricolor coat and floppy ears",
    "Bengal":                   "Bengal cat with spotted rosette coat and bright eyes",
    "Birman":                   "Birman cat with colorpoint coat and white gloves on paws",
    "Bombay":                   "Bombay cat with jet black glossy coat and copper eyes",
    "Boxer":                    "Boxer dog with fawn and white coat and square muzzle",
    "British_Shorthair":        "British Shorthair cat with dense plush blue-grey coat and round chubby face",
    "Chihuahua":                "Chihuahua with tiny body and large upright ears",
    "Egyptian_Mau":             "Egyptian Mau cat with naturally spotted coat and green eyes",
    "English_Cocker_Spaniel":   "English Cocker Spaniel with silky wavy coat and long floppy ears",
    "English_Setter":           "English Setter with speckled belton coat and elegant build",
    "German_Shorthaired":       "German Shorthaired Pointer with liver spotted coat and athletic build",
    "Great_Pyrenees":           "Great Pyrenees with thick white fluffy double coat and majestic build",
    "Havanese":                 "Havanese dog with long silky wavy coat and small compact body",
    "Japanese_Chin":            "Japanese Chin with black and white silky coat and large dark eyes",
    "Keeshond":                 "Keeshond with thick grey and black double coat and spectacles markings",
    "Leonberger":               "Leonberger with lion-like mane and golden brown long coat",
    "Main_Coon":                "Maine Coon cat with large tufted ears and bushy thick shaggy coat",
    "Miniature_Pinscher":       "Miniature Pinscher with sleek rust and black coat and compact muscular build",
    "Newfoundland":             "Newfoundland dog with thick black coat and massive gentle build",
    "Persian":                  "Persian cat with long flowing silky coat and flat face",
    "Pomeranian":               "Pomeranian with fluffy double coat and fox-like face",
    "Pug":                      "Pug with fawn coat and deep wrinkled face and curly tail",
    "Ragdoll":                  "Ragdoll cat with colorpoint semi-long coat and blue eyes",
    "Russian_Blue":             "Russian Blue cat with dense short blue-grey coat and bright green eyes",
    "Saint_Bernard":            "Saint Bernard with red and white thick coat and large gentle face",
    "Samoyed":                  "Samoyed with thick white fluffy double coat and smiling expression",
    "Scottish_Terrier":         "Scottish Terrier with wiry black coat and short legs and pointed ears",
    "Shiba_Inu":                "Shiba Inu with red sesame coat and curled tail and fox-like face",
    "Siamese":                  "Siamese cat with colorpoint short coat and blue almond eyes",
    "Sphynx":                   "Sphynx cat with hairless wrinkled skin and large ears",
    "Staffordshire_Bull_Terrier":"Staffordshire Bull Terrier with smooth brindle coat and muscular build",
    "Wheaten_Terrier":          "Wheaten Terrier with soft silky wheaten coat and square build",
    "Yorkshire_Terrier":        "Yorkshire Terrier with long silky blue and tan coat",
}


# ══════════════════════════════════════════════════════════════════
#  Prompt builder
# ══════════════════════════════════════════════════════════════════

def build_diverse_prompts(breed_key: str, n: int, base_seed: int) -> list[dict]:
    """
    Return n prompts for a breed, each with a unique (pose, location,
    lighting, style) combination drawn deterministically from the pools.

    Returns list of dicts: {prompt, pose, location, lighting, style}
    """
    descriptor = BREED_DESCRIPTORS.get(
        breed_key,
        f"{breed_key.replace('_', ' ')} dog or cat"   # fallback
    )

    # Build a shuffled (but reproducible) list of all combinations
    combos = list(product(
        range(len(POSES)),
        range(len(LOCATIONS)),
        range(len(LIGHTINGS)),
        range(len(STYLES)),
    ))

    # Deterministic shuffle per breed
    import random
    rng = random.Random(base_seed + hash(breed_key) % (2**16))
    rng.shuffle(combos)

    results = []
    for pi, li, lti, si in combos[:n]:
        pose     = POSES[pi]
        location = LOCATIONS[li]
        lighting = LIGHTINGS[lti]
        style    = STYLES[si]
        prompt   = (
            f"a {style} of a {descriptor}, "
            f"{pose}, {location}, {lighting}, "
            f"{QUALITY_SUFFIX}"
        )
        results.append({
            "prompt":   prompt,
            "pose":     pose,
            "location": location,
            "lighting": lighting,
            "style":    style,
        })

    # If n > total combos (unlikely), cycle with varied quality suffix
    while len(results) < n:
        entry = results[len(results) % len(results)].copy()
        entry["prompt"] += ", ultra detailed"
        results.append(entry)

    return results[:n]


# ══════════════════════════════════════════════════════════════════
#  Pipeline loader
# ══════════════════════════════════════════════════════════════════

def build_pipeline(base_model: str, lora_path: str, lora_scale: float, device: str):
    print(f"\n[INFO] Loading base model : {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"[INFO] Loading LoRA weights: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=lora_scale)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xformers enabled")
        except Exception:
            print("[INFO] xformers not available, using default attention")

    return pipe


def class_name_from_dir(dir_name: str) -> str:
    """'15_Abyssinian' -> 'Abyssinian'"""
    parts = dir_name.split("_", 1)
    return parts[1] if len(parts) == 2 else dir_name


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse CLIP-training images for pet classes with LoRA SD 1.5"
    )
    parser.add_argument("--img_dir",          required=True)
    parser.add_argument("--lora_path",        required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--base_model",       default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--images_per_class", type=int,   default=12)
    parser.add_argument("--lora_scale",       type=float, default=0.8)
    parser.add_argument("--steps",            type=int,   default=30)
    parser.add_argument("--guidance",         type=float, default=7.5)
    parser.add_argument("--width",            type=int,   default=224,
                        help="CLIP ViT-B/16 native input size")
    parser.add_argument("--height",           type=int,   default=224)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--save_prompt_log",  action="store_true",
                        help="Save prompts_log.json alongside output_dir")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")
    if device == "cpu":
        print("[WARN] CPU is very slow — GPU strongly recommended.")

    img_dir    = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([
        d for d in img_dir.iterdir()
        if d.is_dir() and re.match(r"^\d+_", d.name)
    ])
    if not class_dirs:
        print(f"[ERROR] No class directories found in {img_dir}")
        return
    print(f"[INFO] Found {len(class_dirs)} classes")

    pipe = build_pipeline(args.base_model, args.lora_path, args.lora_scale, device)

    total_generated = 0
    prompt_log      = {}   # class -> list of prompt metadata

    for idx, class_dir in enumerate(class_dirs, 1):
        class_name = class_name_from_dir(class_dir.name)
        # Match breed key (strip leading number prefix)
        breed_key  = class_name   # e.g. "Abyssinian"

        print(f"\n[{idx:02d}/{len(class_dirs)}] {class_name}")

        class_out = output_dir / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)

        # Resume support
        existing     = list(class_out.glob("*.png"))
        already_done = len(existing)
        remaining    = args.images_per_class - already_done
        if remaining <= 0:
            print(f"    [SKIP] Already complete ({already_done} images).")
            total_generated += already_done
            continue
        if already_done:
            print(f"    [RESUME] {already_done} exist → generating {remaining} more.")

        # Build diverse prompts for ALL images (skip already-done ones)
        all_prompts = build_diverse_prompts(breed_key, args.images_per_class, args.seed)
        prompts_todo = all_prompts[already_done:]   # skip already generated

        prompt_log[class_name] = all_prompts        # log full set

        for i, pmeta in enumerate(prompts_todo):
            img_idx   = already_done + i + 1
            prompt    = pmeta["prompt"]
            seed      = args.seed + idx * 1000 + (already_done + i)
            generator = torch.Generator(device=device).manual_seed(seed)

            print(f"    [{img_idx:02d}/{args.images_per_class}] {prompt[:100]}...")

            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    width=args.width,
                    height=args.height,
                    generator=generator,
                    num_images_per_prompt=1,
                )
                image: Image.Image = result.images[0]
                out_path = class_out / f"{class_dir.name}_{img_idx:03d}.png"
                image.save(out_path)
                print(f"           → saved {out_path.name}  (seed={seed})")
                total_generated += 1

            except Exception as e:
                print(f"    [ERROR] Image {img_idx} failed: {e}")

    # ── Prompt log ──
    if args.save_prompt_log:
        log_path = output_dir / "prompts_log.json"
        log_path.write_text(json.dumps(prompt_log, indent=2, ensure_ascii=False))
        print(f"\n[INFO] Prompt log saved → {log_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"[DONE] {total_generated} images generated.")
    print(f"       Output : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()