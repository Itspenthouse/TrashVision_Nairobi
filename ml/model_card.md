# Model Card: TrashVision Custom Detector

## Overview

| Field | Value |
|---|---|
| Version | `trashvision-custom-v1` |
| Architecture | YOLOv8 nano transfer learning |
| Task | Object detection |
| Classes | `organic_waste`, `drain_blockage`, `standing_water` |
| Intended use | Human-reviewed prioritization of visible market conditions |

The model is decision support. It must not automatically dispatch cleanup
teams, make enforcement decisions, or be described as flood forecasting.

## Dataset

The initial public-data pipeline uses the MIT-licensed TACO dataset. Only TACO
annotations named `Food waste` map to `organic_waste`. Other TACO litter
categories are excluded rather than mislabeled.

TACO has no ground-truth classes for `drain_blockage` or `standing_water`.
Those classes require consented, locally relevant, manually reviewed field
images before the checkpoint can claim three-class coverage.

The preparation script uses a deterministic 70/20/10 image split. Source:
https://github.com/pedropro/TACO

## Metrics

Training must populate this section from the held-out test set before a model
is promoted:

| Metric | Current value |
|---|---|
| mAP50 | Not yet measured |
| mAP50-95 | Not yet measured |
| Per-class precision | Not yet measured |
| Per-class recall | Not yet measured |

A checkpoint without these measurements is experimental.

## Known limitations

- Public training data is not Nairobi-market specific.
- Drain blockage and standing water currently lack public training coverage.
- Faces, plates, poor lighting, occlusion, and camera motion can reduce quality.
- The severity score is a deterministic visible-condition score, not a learned
  estimate of operational impact.
- Low-confidence and empty results require human review.
