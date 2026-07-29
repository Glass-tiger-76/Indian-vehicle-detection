import gradio as gr
from ultralytics import YOLO

model = YOLO('best.pt')

def detect(image):
    results = model(image)
    annotated = results[0].plot()          
    
    counts = {}
    for box in results[0].boxes:
        name = model.names[int(box.cls)]
        counts[name] = counts.get(name, 0) + 1
    
    return annotated, counts

gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Detections"), gr.JSON(label="Vehicle counts")],
).launch()
