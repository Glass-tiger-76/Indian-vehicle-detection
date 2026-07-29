# Indian Road Vehicle Detection (YOLOv8)

Object detection for Indian traffic scenes. COCO-pretrained detectors miss most of what's actually on Indian roads — auto-rickshaws, totos, e-rickshaws, cycle-rickshaws, tempos — so this fine-tunes YOLOv8 on a 14-class Indian vehicle dataset.

**mAP50: 0.935** | **mAP50-95: 0.810** across 14 classes.

## Dataset

[IRUVD](https://www.kaggle.com/datasets/asfakali2/iruvd-dataset-for-automatic-vehicle-detection) — Indian Road User Vehicle Detection.

- 3,200 training images / 800 validation images
- 2,877 labelled instances in validation
- 14 classes: truck, cyclist, bike, tempo, car, zeep, toto, e-rickshaw, auto-rickshaw, bus, van, cycle-rickshaw, person, taxi

## Training

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
)
```

Transfer learning from COCO weights. Trained on a single T4.

## Results

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| e-rickshaw | 0.952 | 0.981 | 0.987 |
| truck | 0.959 | 0.968 | 0.986 |
| toto | 0.965 | 0.970 | 0.983 |
| bus | 0.914 | 0.932 | 0.972 |
| van | 0.933 | 0.939 | 0.956 |
| tempo | 0.909 | 0.896 | 0.953 |
| cycle-rickshaw | 0.962 | 0.940 | 0.953 |
| car | 0.947 | 0.856 | 0.945 |
| bike | 0.953 | 0.881 | 0.941 |
| person | 0.929 | 0.850 | 0.933 |
| zeep | 0.861 | 0.865 | 0.912 |
| auto-rickshaw | 0.927 | 0.812 | 0.905 |
| cyclist | 0.912 | 0.765 | 0.855 |
| taxi | 0.787 | 0.801 | 0.807 |
| **all** | **0.922** | **0.890** | **0.935** |

## What the errors look like

The weak classes are not random. Every class that underperforms is one that **visually overlaps with another class in the set**:

- **taxi (0.807)** — weakest on both precision and recall. Taxis are cars. The only distinguishing features are livery and markings, which are small at 640px and inconsistent across vehicles.
- **cyclist (0.855, recall 0.765)** — competes with `bike` and `person`. Thin, small, and the person-on-bicycle boundary is genuinely ambiguous.
- **auto-rickshaw (0.905, recall 0.812)** — competes with `toto` and `e-rickshaw`, both three-wheelers with similar silhouettes.

Meanwhile the classes with distinctive shapes — trucks, buses, vans, e-rickshaws — all sit above 0.95. The model detects *shape* reliably and struggles with *fine-grained distinctions between similar shapes*.

This suggests the ceiling here is a labelling/data problem rather than a capacity problem: more instances of the confusable classes would help more than a larger backbone.

These were clearly demonstrated by the Annotated example below

## Limitations

- Validated only on IRUVD's distribution. Performance on other camera angles, lighting, or regions is unmeasured.
- `recall 0.81` on auto-rickshaws means roughly 1 in 5 is missed. For counting applications this biases counts downward.
- No density/occupancy layer yet — the detector outputs boxes, nothing downstream consumes them.

## Next

- Occupancy-ratio density estimation from box areas (thresholds need calibrating against real scenes, not guessed)
- More instances for taxi / cyclist / auto-rickshaw specifically

## Files

- `train.py` — training script
- `app.py` — Gradio demo (upload image → annotated output + per-class counts)
- `weights/best.pt` — trained weights

# Annotated Example 


<img width="3008" height="2000" alt="Karol_Bagh,_2008_(14)" src="https://github.com/user-attachments/assets/b764546c-2e10-4f33-946f-9d5f313edaea" />



<img width="3008" height="2000" alt="image" src="https://github.com/user-attachments/assets/c8bf719f-af9f-478b-8cc2-fd6b0ee6d6cf" />

```python
{
  "auto-rickshaw": 4,
  "bike": 7,
  "tempo": 1,
  "toto": 1,
  "bus": 1,
  "cyclist": 1
}
```
