# S3 YOLO

S3 YOLO is a YOLO-based object detection project with training, detection, and validation entry points adapted for this repository.

Use:

- `global_train.py` to train or fine-tune a model.
- `s3_detect.py` to run inference on images, videos, streams, screenshots, directories, or `.npy` inputs.
- `s3_val.py` to evaluate a trained model on a dataset split.

## Setup

Create and activate a Python environment, then install the project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requ_py.txt
```

If `requ_py.txt` is too strict for your machine, install the core package metadata instead:

```bash
pip install -e .
```

For GPU runs, make sure your installed PyTorch build matches your CUDA version.

## Dataset Configuration

Training and validation use YOLO dataset YAML files. The repository includes examples in `data/`, such as:

- `data/etram_npy.yaml`
- `data/etram_day.yaml`
- `data/coco128.yaml`

A dataset YAML should define the train/validation paths, number of classes, and class names. For example:

```yaml
train: /path/to/train/images
val: /path/to/val/images

nc: 1
names: ["object"]
```

Use the dataset YAML path with `--data`.

## Model Configuration

Model YAML files are stored in `models/`. Common choices include:

- `models/S3_YOLO_s.yaml`
- `models/S3_YOLO_float.yaml`
- `models/S3_YOLO_4bits.yaml`
- `models/S3_YOLO_unified.yaml`
- `models/LoLa_YOLO.yaml`

Use the model YAML path with `--cfg`.

## Training

Train with `global_train.py`.

Basic training command:

```bash
python global_train.py \
  --data data/etram_npy.yaml \
  --cfg models/S3_YOLO_s.yaml \
  --weights "" \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --device 0
```

Fine-tune from existing weights:

```bash
python global_train.py \
  --data data/etram_npy.yaml \
  --cfg models/S3_YOLO_s.yaml \
  --weights path/to/checkpoint.pt \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --device 0
```

Train a quantized model:

```bash
python global_train.py \
  --data data/etram_npy.yaml \
  --cfg models/S3_YOLO_4bits.yaml \
  --weights "" \
  --quantize \
  --bits 4 \
  --first-layer-bits 8 \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --device 0
```

Resume the latest run:

```bash
python global_train.py --resume
```

Training outputs are saved under `runs/train/` by default. The best and last checkpoints are usually written inside the run's `weights/` folder.

Useful training options:

- `--data`: dataset YAML path.
- `--cfg`: model YAML path.
- `--weights`: initial checkpoint path, or `""` to train from scratch.
- `--epochs`: number of epochs.
- `--batch-size`: total batch size.
- `--imgsz`: training and validation image size.
- `--device`: CUDA device, such as `0`, `0,1`, or `cpu`.
- `--hyp`: hyperparameter YAML path.
- `--project`: output directory, default `runs/train`.
- `--name`: run name.
- `--exist-ok`: reuse an existing output folder.
- `--quantize`: build the quantized model variant.
- `--bits`: quantized layer bit width.
- `--first-layer-bits`: first quantized layer bit width.

## Detection

Run detection with `s3_detect.py`.

Basic command:

```bash
python s3_detect.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --source path/to/images_or_video \
  --imgsz 640 \
  --conf-thres 0.5 \
  --device 0
```

Run on a single image:

```bash
python s3_detect.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --source path/to/image.jpg
```

Run on a directory:

```bash
python s3_detect.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --source path/to/images/
```

Run on `.npy` event input:

```bash
python s3_detect.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --source path/to/sample.npy
```

Save YOLO-format text predictions:

```bash
python s3_detect.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --source path/to/images/ \
  --save-txt \
  --save-conf
```

Detection outputs are saved under `runs/detect/` by default.

Useful detection options:

- `--weights`: trained checkpoint path.
- `--cfg`: model YAML path.
- `--data`: dataset YAML path with class names.
- `--source`: input file, folder, URL, webcam index, screenshot, stream, or `.npy` file.
- `--imgsz`: inference size. Use one value for square inference or two values for height and width.
- `--conf-thres`: confidence threshold.
- `--iou-thres`: NMS IoU threshold.
- `--save-txt`: save predictions as YOLO text labels.
- `--save-conf`: include confidence scores in saved labels.
- `--save-crop`: save cropped detections.
- `--nosave`: do not save annotated images or videos.
- `--classes`: filter by class index.
- `--project`: output directory, default `runs/detect`.
- `--name`: run name.

## Validation

Evaluate a checkpoint with `s3_val.py`.

Basic validation command:

```bash
python s3_val.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --batch-size 32 \
  --imgsz 640 \
  --device 0
```

Show per-class metrics:

```bash
python s3_val.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --verbose
```

Save validation predictions as labels:

```bash
python s3_val.py \
  --weights runs/train/exp/weights/best.pt \
  --cfg models/S3_YOLO_s.yaml \
  --data data/etram_npy.yaml \
  --save-txt \
  --save-conf
```

Validation outputs are saved under `runs/val/` by default. The script reports precision, recall, mAP50, and mAP50-95.

Useful validation options:

- `--data`: dataset YAML path.
- `--weights`: trained checkpoint path.
- `--cfg`: model YAML path.
- `--batch-size`: validation batch size.
- `--imgsz`: inference size.
- `--conf-thres`: confidence threshold.
- `--iou-thres`: NMS IoU threshold.
- `--task`: dataset split, usually `val`, `test`, or `train`.
- `--device`: CUDA device or `cpu`.
- `--verbose`: print per-class results.
- `--save-txt`: save predictions as YOLO text labels.
- `--save-json`: save COCO-style JSON predictions.
- `--project`: output directory, default `runs/val`.
- `--name`: run name.

## Typical Workflow

1. Prepare a dataset YAML in `data/`.
2. Pick a model YAML from `models/`.
3. Train with `global_train.py`.
4. Validate the best checkpoint with `s3_val.py`.
5. Run detection with `s3_detect.py`.

Example:

```bash
python global_train.py --data data/etram_npy.yaml --cfg models/S3_YOLO_s.yaml --weights "" --epochs 100 --device 0
python s3_val.py --data data/etram_npy.yaml --cfg models/S3_YOLO_s.yaml --weights runs/train/exp/weights/best.pt --device 0
python s3_detect.py --data data/etram_npy.yaml --cfg models/S3_YOLO_s.yaml --weights runs/train/exp/weights/best.pt --source path/to/input --device 0
```

## Notes

- Generated training, validation, and detection outputs are written to `runs/`.
- Large model weights, ONNX files, FPGA build outputs, images, videos, and datasets are intentionally ignored by Git.
- For CPU-only runs, replace `--device 0` with `--device cpu`.
- If you change the number of classes, update both the dataset YAML and the model/checkpoint combination accordingly.
