# Custom RGB + Optical Flow Dataloader Integration for YOLOv11

This guide turns the high-level repository walkthrough into an actionable implementation plan for adding a custom multi-modal dataloader that serves both object detection and instance segmentation training in Ultralytics YOLO (targeting the v11 development branch). The workflow assumes COCO JSON or YOLO text annotations with bounding boxes, class labels, and optional segmentation masks, and that RGB frames and optical-flow maps are stored locally in a mirrored directory structure.

## 1. Define Dataset Contracts

1. **Decide on task separation** – Keep detection and segmentation training modes distinct, matching the existing trainer specializations. Detection uses `DetectionTrainer`, while segmentation extends it via `SegmentationTrainer`.【F:ultralytics/models/yolo/detect/train.py†L20-L151】【F:ultralytics/models/yolo/segment/train.py†L13-L123】
2. **Set channel expectations** – YOLO datasets propagate the expected channel count via `data["channels"]`. Ensure the dataset YAML advertises `channels: 6` (RGB + flow) so downstream trainers initialise models with the correct input dimensions.【F:ultralytics/data/dataset.py†L73-L89】【F:ultralytics/models/yolo/detect/train.py†L129-L144】【F:ultralytics/models/yolo/segment/train.py†L53-L76】
3. **Annotation formats** – Confirm each sample has:
   - RGB image path
   - Optical flow tensor/image path (e.g., encoded as 3 channels for dx/dy/magnitude or 2 channels + 1 padding)
   - Detection labels (YOLO txt or COCO JSON)
   - Segmentation masks where required
4. **Directory layout** – Mirror RGB and flow trees so that simple path substitution (e.g., replacing `/rgb/` with `/flow/`) retrieves the paired flow file. Log the mapping logic in your dataset class.

## 2. Extend the Dataset Layer

1. **Create a dedicated dataset class** – Derive `RGBFlowDataset` from `YOLODataset` to reuse caching, label parsing, and augmentation pipelines.【F:ultralytics/data/dataset.py†L47-L200】
   - Override `load_image`/`get_image` hooks (or the point where images enter the transform stack) to read both RGB and flow, concatenate along channel dimension, and update metadata (`samples[i]["img"]`).
   - Maintain segmentation compatibility by keeping mask loading untouched; `YOLODataset` already handles segments when `task == "segment"` via `self.use_segments`.
2. **Augment transforms** – Call `super().build_transforms()` then append any flow-specific preprocessing (normalisation, optional optical-flow augmentations).【F:ultralytics/data/dataset.py†L317-L385】
   - Respect current defaults (Mosaic, MixUp) and defer advanced flow augmentations until explicitly requested.
3. **Collate adjustments** – If flow tensors require special batching (e.g., lower precision), adjust or override `collate_fn`. Base implementation already stacks tensors and concatenates masks/boxes.【F:ultralytics/data/dataset.py†L294-L314】
4. **Caching metadata** – Reuse the label cache mechanism. Add flow file hashes to the cache to detect stale pairs if needed.【F:ultralytics/data/dataset.py†L90-L200】

## 3. Wire the Builder Helpers

1. **Expose the dataset** – Register `RGBFlowDataset` in `ultralytics/data/__init__.py` for external imports.
2. **Flag selection** – Extend `build_yolo_dataset` to choose your dataset when a new argument (e.g., `multi_modal="rgb_flow"`) or configuration flag is passed. The builder currently toggles between `YOLODataset` and `YOLOMultiModalDataset`; mirror that logic for your class.【F:ultralytics/data/build.py†L114-L133】
3. **Pass-through metadata** – Ensure the builder forwards:
   - `channels=cfg.channels` (already done through `cfg`)
   - Any custom kwargs (e.g., `flow_suffix`, `flow_format`).
4. **CLI/Config plumbing** – Update the task overrides (YAML or argparse) so users can enable the new loader via `--multi-modal rgb_flow` or similar.

## 4. Trainer Integration Tasks

1. **Detection** – `DetectionTrainer.build_dataset()` already delegates to `build_yolo_dataset`. Add logic to set your selection flag based on new CLI args or `data.yaml` entries.【F:ultralytics/models/yolo/detect/train.py†L53-L90】
2. **Segmentation** – Because `SegmentationTrainer` inherits detection behaviour after forcing `task="segment"`, the same dataset plumbing will apply once the detection trainer recognises the flag.【F:ultralytics/models/yolo/segment/train.py†L30-L83】
3. **Batch preprocessing** – Confirm `preprocess_batch` normalises the expanded channel tensors correctly. Default behaviour simply divides by 255, which still holds if the flow channels are stored in 0–255 or pre-scaled ranges.【F:ultralytics/models/yolo/detect/train.py†L91-L116】
4. **Model configs** – Update or clone the YOLOv11 model YAMLs (`yolo11n.yaml`, `yolo11n-seg.yaml`, etc.) to set the input channel count and, if necessary, adapt the first convolution kernel.

## 5. Configuration and YAML Updates

1. **Dataset YAML** – Document expected keys: `path`, `train`, `val`, optional `test`, `names`, `channels: 6`, plus custom keys (`flow_suffix`, `flow_format`).
2. **Default overrides** – Introduce defaults in `cfg/defaults.py` so experiments can enable the dataloader without manual edits.
3. **Segmentation masks** – Clarify mask storage (PNG per-instance or COCO RLE) and ensure the dataset class normalises them to the format `YOLODataset` expects (`segments` arrays or binary masks).

## 6. Validation & Testing Checklist

1. **Unit tests** – Add coverage under `tests/` to instantiate the dataset with synthetic RGB + flow pairs, verifying shape `[B, 6, H, W]` and mask integrity.
2. **Smoke training** – Run short detection and segmentation epochs using the CLI to confirm end-to-end compatibility:
   - `yolo detect train data=rgb_flow.yaml model=yolo11n.yaml imgsz=640 epochs=1`
   - `yolo segment train data=rgb_flow-seg.yaml model=yolo11n-seg.yaml imgsz=640 epochs=1`
3. **Export sanity** – If exporting models, ensure the additional channels propagate to exporters (ONNX/TensorRT). Update exporter configs if required.

## 7. Operational FAQ

| Question | Guidance |
| --- | --- |
| **Which YOLO release should be targeted first?** | Begin with the YOLOv11 development branch to match the latest Ultralytics interfaces. |
| **Can detection and segmentation run simultaneously?** | Keep them as separate training runs, reusing the same dataset class but with `task` overrides, mirroring existing protocol. |
| **Do we support RGB-only or flow-only fallbacks?** | Implement graceful fallbacks: if a paired file is missing, either drop the sample (strict mode) or substitute zeros/replicated channels, controlled via a config flag. |
| **How should inputs be normalised?** | Follow Ultralytics defaults for RGB (divide by 255) and apply any flow-specific normalisation inside the dataset transforms so downstream trainers remain unchanged. |

## 8. Next Steps

1. Scaffold the new dataset class and builder flag, then run unit tests.
2. Update configs and documentation so the dataloader can be toggled from the CLI.
3. Iterate with sample training runs, profiling throughput and ensuring compatibility with later augmentation enhancements.

With this roadmap you can incrementally deliver a production-ready RGB + optical flow dataloader that integrates cleanly with YOLOv11 detection and segmentation workflows.
