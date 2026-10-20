"""
Script to generate a 1,000-photo realistic test dataset for AI Family Photo Sorter.
- 50 images of Target Person (derived with realistic variations from target face)
- 450 images of Distractor Person (different person with real face)
- 500 images of wedding venue, decor, florals, and festive party backgrounds (non-face)
- 1 clean reference photo of Target Person in a separate folder
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

def create_dataset():
    base_dir = r"C:\AI_PHOTO FINDER\test_dataset"
    source_dir = os.path.join(base_dir, "source_photos")
    ref_dir = os.path.join(base_dir, "reference_photo")
    dest_high = os.path.join(base_dir, "dest_high")
    dest_review = os.path.join(base_dir, "dest_review")

    for d in [source_dir, ref_dir, dest_high, dest_review]:
        os.makedirs(d, exist_ok=True)

    target_face_path = r"C:\AI_PHOTO FINDER\reference_photos\me_52da70f9_IMG-20251011-WA0019.jpg"
    distractor_face_path = r"C:\AI_PHOTO FINDER\reference_photos\me_4e6ff4ed_WIN_20211004_08_16_14_Pro.jpg"

    # 1. Save clean reference photo
    target_pil = Image.open(target_face_path).convert("RGB")
    target_ref_save_path = os.path.join(ref_dir, "target_person_reference.jpg")
    target_pil.save(target_ref_save_path, "JPEG", quality=95)
    print(f"Reference photo saved to: {target_ref_save_path}")

    distractor_pil = Image.open(distractor_face_path).convert("RGB")

    # Pick 50 random slots out of 1000 for the target person
    random.seed(42)
    all_indices = list(range(1, 1001))
    target_indices = set(random.sample(all_indices, 50))

    # Remaining 950: 450 distractor faces, 500 venue/decor/party scenes
    remaining_indices = [idx for idx in all_indices if idx not in target_indices]
    distractor_face_indices = set(random.sample(remaining_indices, 450))
    scenery_indices = set([idx for idx in remaining_indices if idx not in distractor_face_indices])

    manifest = {
        "dataset_name": "AI Photo Sorter 1000-Photo Benchmark Dataset",
        "total_images": 1000,
        "target_person_images_count": 50,
        "target_person_indices": sorted(list(target_indices)),
        "distractor_face_images_count": len(distractor_face_indices),
        "venue_and_decor_images_count": len(scenery_indices),
        "reference_photo_path": target_ref_save_path,
        "source_photos_dir": source_dir,
        "dest_high_dir": dest_high,
        "dest_review_dir": dest_review,
        "items": {}
    }

    print("Generating 1,000 photos...")

    def augment_face(img, seed):
        rng = random.Random(seed)
        w, h = img.size

        # Mild crop / zoom (82% to 98%)
        zoom = rng.uniform(0.82, 0.98)
        cw, ch = int(w * zoom), int(h * zoom)
        x0 = rng.randint(0, w - cw)
        y0 = rng.randint(0, h - ch)
        cropped = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.LANCZOS)

        # Mild brightness & contrast
        b_factor = rng.uniform(0.88, 1.15)
        c_factor = rng.uniform(0.90, 1.12)
        res = ImageEnhance.Brightness(cropped).enhance(b_factor)
        res = ImageEnhance.Contrast(res).enhance(c_factor)

        # Mild color adjustment
        color_factor = rng.uniform(0.92, 1.08)
        res = ImageEnhance.Color(res).enhance(color_factor)

        # Mild rotation (-3.0 to +3.0 degrees)
        angle = rng.uniform(-3.0, 3.0)
        res = res.rotate(angle, resample=Image.Resampling.BILINEAR)

        # Quality
        q = rng.randint(82, 95)
        return res, q

    # Color palettes for realistic wedding/party backgrounds
    palettes = [
        [(245, 230, 210), (180, 140, 90), (220, 190, 150)],  # Gold / Champagne
        [(230, 210, 220), (140, 80, 110), (210, 160, 180)],  # Rose / Floral
        [(210, 230, 240), (70, 110, 150), (160, 195, 220)],   # Evening blue
        [(240, 235, 225), (100, 130, 90), (180, 205, 170)],  # Garden / Botanical
        [(30, 25, 40), (80, 60, 100), (210, 180, 120)],       # Festive banquet lights
        [(250, 245, 240), (215, 190, 160), (240, 225, 210)], # Cake / Banquet table
    ]

    def generate_scene(seed):
        rng = random.Random(seed)
        palette = rng.choice(palettes)
        w, h = 800, 600

        # Create gradient / backdrop
        base_color = palette[0]
        grad_color = palette[1]
        accent_color = palette[2]

        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            alpha = y / h
            r = int(base_color[0] * (1 - alpha) + grad_color[0] * alpha)
            g = int(base_color[1] * (1 - alpha) + grad_color[1] * alpha)
            b = int(base_color[2] * (1 - alpha) + grad_color[2] * alpha)
            arr[y, :] = [r, g, b]

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Draw decorative party bokeh circles / floral shapes
        for _ in range(rng.randint(15, 30)):
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            rad = rng.randint(20, 90)
            fill_c = (
                rng.randint(min(base_color[0], accent_color[0]), 255),
                rng.randint(min(base_color[1], accent_color[1]), 255),
                rng.randint(min(base_color[2], accent_color[2]), 255),
            )
            draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=fill_c)

        img = img.filter(ImageFilter.GaussianBlur(radius=rng.randint(8, 20)))
        q = rng.randint(80, 92)
        return img, q

    # Generate all 1,000 photos
    for idx in range(1, 1001):
        filename = f"photo_{idx:04d}.jpg"
        filepath = os.path.join(source_dir, filename)

        if idx in target_indices:
            img, quality = augment_face(target_pil, seed=idx * 7)
            img.save(filepath, "JPEG", quality=quality)
            manifest["items"][filename] = {
                "type": "target_person",
                "expected": "high_confidence",
                "is_target": True,
            }
        elif idx in distractor_face_indices:
            img, quality = augment_face(distractor_pil, seed=idx * 13)
            img.save(filepath, "JPEG", quality=quality)
            manifest["items"][filename] = {
                "type": "distractor_face",
                "expected": "no_match",
                "is_target": False,
            }
        else:
            img, quality = generate_scene(seed=idx * 17)
            img.save(filepath, "JPEG", quality=quality)
            manifest["items"][filename] = {
                "type": "venue_and_decor",
                "expected": "no_match",
                "is_target": False,
            }

        if idx % 100 == 0:
            print(f"Generated {idx} of 1,000 photos...")

    manifest_path = os.path.join(base_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] 1,000 photos successfully created!")
    print(f"Source folder: {source_dir}")
    print(f"Reference folder: {ref_dir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    create_dataset()
