import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np

model_path = "pretrained/mask2former/mask2former_swinl_coco_pano"

# load Mask2Former fine-tuned on COCO panoptic segmentation
processor = AutoImageProcessor.from_pretrained(model_path)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("demos/1682391069844152.png")
inputs = processor(images=image, return_tensors="pt", size=800)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits
print(outputs.keys())
print(outputs.pixel_decoder_last_hidden_state.shape)

# you can pass them to processor for postprocessing
result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
predicted_panoptic_map = result["segmentation"]
obj_ids = np.unique(predicted_panoptic_map.numpy())
print(obj_ids)

colors = [
    # (255, 0, 0), (0, 255, 0), (0, 0, 255),
    #           (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255), [255, 192, 203],  # Pink
              [165, 42, 42],    # Brown
              [255, 165, 0],    # Orange
              [128, 0, 128],     # Purple
              [0, 0, 128],       # Navy
              [128, 0, 0],      # Maroon
              [128, 128, 0],    # Olive
              [70, 130, 180],   # Steel Blue
              [173, 216, 230],  # Light Blue
              [255, 192, 0],    # Gold
              [255, 165, 165],  # Light Salmon
              [255, 20, 147],   # Deep Pink
              ]

image.save("./mask2former_image_demo.png")
mask = np.zeros(list(predicted_panoptic_map.shape) + [3]).astype(np.uint8)
for i, obj_id in enumerate(obj_ids[:1]):
    mask[predicted_panoptic_map==obj_id] = np.array(colors[i])
image = np.array(image)
mask = (image * 0.5 + mask * 0.5).astype(np.uint8)
mask = Image.fromarray(mask)
mask.save("./mask2former_mask.png")

