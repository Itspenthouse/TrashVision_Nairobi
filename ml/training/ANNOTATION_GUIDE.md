# Annotation Guide — TrashVision Nairobi

The shared rulebook for labeling images. If everyone labels the same way, the
model learns cleanly. If people label differently, the model gets confused. This
guide is the "one shared label guide" the SDD (4.2) requires. Read it before
drawing a single box.

---

## The 3 classes (and ONLY these 3)

| id | class | Draw a box around... | Do NOT box |
|----|-------|----------------------|------------|
| 0 | `organic_waste` | Piles/scatter of food scraps, produce, market organic refuse (peels, rotting veg, food waste heaps). | Clean packaged goods for sale; people; vehicles. |
| 1 | `drain_blockage` | A drain/gutter/channel visibly clogged with waste, silt, or debris so water can't pass. | An open, clean, flowing drain. |
| 2 | `standing_water` | Pooled/stagnant water sitting on ground, in a gutter, or a puddle that isn't draining. | Clean flowing water; wet-but-not-pooled ground; deliberate water containers. |

> The `id` is what goes in the label file. Order matters — it must match
> `data.yaml` (0=organic_waste, 1=drain_blockage, 2=standing_water).

---

## How to draw boxes (the golden rules)

1. **Tight, not loose.** The box should hug the object — no big margin of
   background, but don't clip the object either.
2. **One box per distinct object/area.** A big waste pile = one box. Three
   separate puddles = three boxes.
3. **Occlusion is fine.** If a drain is half-hidden by trash, box the whole drain
   area you can reasonably infer.
4. **Ambiguous? Skip or flag it.** If you genuinely can't tell, don't guess —
   mislabels hurt more than missing labels. Set it aside for peer review.
5. **Label everything in-frame** that fits a class. Don't leave a clear waste
   pile unlabeled just because there's already one in the image.

---

## Tools (pick one, all export YOLO format)

- **Roboflow** (web, easiest, has a free tier) — draw boxes, export "YOLOv8".
- **LabelImg** (desktop, offline) — set format to "YOLO".
- **CVAT** (web, powerful, good for teams).

All of them output the exact format below, so you never write it by hand.

## The label file format (what the tool produces)

For an image `market_001.jpg`, the tool writes `market_001.txt`:

```
<class_id> <x_center> <y_center> <width> <height>
```

- One line per box.
- All 4 numbers are **fractions of image size (0.0–1.0)**, not pixels.
- Example — one organic_waste pile centered slightly left, medium size:
  ```
  0 0.42 0.55 0.30 0.25
  ```
- An image with **no** target objects gets an **empty** `.txt` (this is a
  valuable "negative" example — keep some!).

Images go in `dataset/images/{train,val,test}/`, labels in the matching
`dataset/labels/{train,val,test}/` with the **same filename**.

---

## Dataset quality checklist (SDD 4.2)

- [ ] 150–300 images total, **roughly balanced** across the 3 classes.
- [ ] Variety: different markets, lighting (morning/noon/overcast), angles,
      distances, and some **clean negatives** (no waste).
- [ ] **70/20/10** split into train/val/test. Keep near-duplicate photos or the
      same scene in the **same** split (prevents "leakage" that fakes good scores).
- [ ] At least **20%** of labels peer-checked by a second person.
- [ ] **Privacy:** no faces, no vehicle plates, no personal details. Blur or
      exclude. Use only consented/licensed images. Delete demo data after the event.

---

## Common mistakes that quietly ruin a model

- Boxing the *whole image* instead of the object.
- Inconsistent decisions ("is a wet floor standing_water?") — agree as a team ONCE.
- Putting the same photo in both train and test (leakage → dishonest scores).
- Forgetting negatives → the model thinks *everything* is waste.
