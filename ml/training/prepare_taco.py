"""Download licensed TACO images and convert food-waste boxes to YOLO format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
FOOD_CATEGORY_NAMES = {"Food waste"}
HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "TrashVision/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def split_for(image_id: int, seed: int) -> str:
    randomizer = random.Random(f"{seed}:{image_id}")
    value = randomizer.random()
    if value < 0.7:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def prepare(limit: int | None, seed: int) -> None:
    annotation_file = HERE / "taco_annotations.json"
    if not annotation_file.exists():
        print("Downloading official TACO annotations...")
        download(ANNOTATIONS_URL, annotation_file)

    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in payload["categories"]}
    food_ids = {
        category_id
        for category_id, name in category_names.items()
        if name in FOOD_CATEGORY_NAMES
    }
    if not food_ids:
        available = ", ".join(sorted(category_names.values()))
        raise RuntimeError(
            f"TACO does not contain {sorted(FOOD_CATEGORY_NAMES)}. Available: {available}"
        )

    annotations = defaultdict(list)
    for item in payload["annotations"]:
        if item["category_id"] in food_ids:
            annotations[item["image_id"]].append(item)

    images = [item for item in payload["images"] if item["id"] in annotations]
    random.Random(seed).shuffle(images)
    if limit:
        images = images[:limit]

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    counts = Counter()
    for index, image in enumerate(images, start=1):
        split = split_for(image["id"], seed)
        suffix = Path(image["file_name"]).suffix.lower() or ".jpg"
        stem = f"taco_{image['id']:05d}"
        image_path = DATASET / "images" / split / f"{stem}{suffix}"
        source_url = image.get("flickr_url") or image.get("coco_url")
        if not source_url:
            print(f"Skipping image {image['id']}: no source URL")
            continue
        try:
            download(source_url, image_path)
        except Exception as exc:
            print(f"Skipping image {image['id']}: {exc}")
            continue

        width, height = image["width"], image["height"]
        rows = []
        for annotation in annotations[image["id"]]:
            x, y, box_width, box_height = annotation["bbox"]
            rows.append(
                "0 "
                f"{(x + box_width / 2) / width:.6f} "
                f"{(y + box_height / 2) / height:.6f} "
                f"{box_width / width:.6f} "
                f"{box_height / height:.6f}"
            )
        (DATASET / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="ascii"
        )
        counts[split] += 1
        print(f"[{index}/{len(images)}] {split}: {image_path.name}")

    print(f"Prepared {sum(counts.values())} real images: {dict(counts)}")
    print("Coverage: organic_waste only. Drain and water need labeled field images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Optional image limit for a short run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.limit, args.seed)
