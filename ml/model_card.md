# Model Card — TrashVision Nairobi (ML lane)

A model card is a short, honest "nutrition label" for an AI model. The SDD
(section 4.4) requires one. It tells anyone using the model what it is, how it
was built, how well it works, and — most importantly — where it fails. Keep this
file updated as the model changes; it's a core demo-honesty artifact.

---

## 1. Model overview

| | |
|---|---|
| **Name** | TrashVision detector |
| **Current version** | `pretrained-yolov8n-coco-proxy-v0` |
| **Task** | Object detection (draw boxes around waste-related objects) |
| **Architecture** | YOLOv8-nano (smallest YOLOv8), run on CPU |
| **Target classes** | `organic_waste`, `drain_blockage`, `standing_water` |
| **Intended use** | Decision *support* for market cleanup coordinators — suggest where visible problems are. NOT an authority. |
| **Out of scope** | Automatic dispatch, legal/enforcement decisions, validated flood forecasting, medical/safety guarantees. |

---

## 2. Two modes (be explicit about which one is running)

### a) Demo / fallback mode — CURRENT (`use_coco_proxy = true`)
- Uses the **stock pretrained YOLOv8n** trained on COCO (80 everyday objects).
- COCO **food** classes (banana, apple, pizza, …) are mapped to `organic_waste`
  as an honest **proxy**, because a market waste pile genuinely contains these.
- **Known limitation, stated plainly:** the pretrained model has **no reliable
  signal for `drain_blockage` or `standing_water`.** In demo mode those classes
  will almost always score 0. This is expected, and it is exactly *why* the
  custom dataset (below) matters.

### b) Custom mode — FUTURE (`use_coco_proxy = false`)
- A YOLOv8n **fine-tuned** on our own labeled images, outputting the 3 real
  classes directly. Trained via `training/train.py`. Fill in metrics below once
  it exists.

---

## 3. Dataset (custom model)

| Item | Plan (from SDD 4.2) | Status |
|---|---|---|
| Source | 150–300 **consented/licensed** photos across all 3 classes, varied lighting/angle/market | ☐ to collect |
| Annotation | Bounding boxes, one shared label guide, ≥20% peer-checked | ☐ |
| Split | 70% train / 20% val / 10% test; keep near-duplicates in the SAME split (avoid leakage) | ☐ |
| Privacy | No faces, plates, or personal details; delete demo data after the event | ☐ |

---

## 4. Scoring (deterministic, not learned)

The 0–100 severity score is **not** produced by the neural network — it's a
fixed formula in `app/scoring.py`, so it's fully explainable and testable:

```
severity   = 0.45*organic_waste + 0.25*drain_blockage + 0.20*standing_water + 0.10*confidence
risk_proxy = 0.55*drain_blockage + 0.35*standing_water + 0.10*severity      (visible-condition proxy ONLY)
```
Priority bands: 0–29 low · 30–54 medium · 55–79 high · 80–100 urgent.

`risk_proxy` is a **visible-condition proxy, not a weather forecast.** Never
present it as flood prediction.

---

## 5. Metrics

Fill in after training the custom model (`python training/train.py` prints these):

| Metric | Meaning | Value |
|---|---|---|
| mAP50 | avg precision at a lenient box overlap | _TBD_ |
| mAP50-95 | stricter averaged precision (the honest number) | _TBD_ |
| Per-class recall | of real X, how many did we catch? | _TBD_ |
| Confusion notes | which classes get mixed up | _TBD_ |

_Demo mode has no meaningful accuracy metric — it's a proxy, so we report the
limitation instead of a number._

---

## 6. Failure modes (always show at least one in the demo)

- **Blockage/water in demo mode:** effectively undetectable → correctly flagged
  `needs_review`. Good example of the responsible-AI path working.
- **Cluttered scenes:** overlapping objects can be missed or double-counted.
- **Unusual lighting / odd angles / motion blur:** confidence drops → review.
- **Look-alikes:** e.g. a reflective wet floor vs standing water.

When confidence is below the review threshold, the system says *"AI suggestion,
flagged for human review"* — never *"this IS ..."*.

---

## 7. Responsible-AI checklist (SDD 4.4)

- [x] Confidence + model version shown with every prediction.
- [x] Wording is "AI suggests", not "AI certainty".
- [x] Low-confidence / empty results routed to `needs_review`.
- [x] No names or phone numbers collected by this service.
- [x] Users warned (in the app) not to photograph faces/plates/personal details.
- [x] This model card kept up to date.
