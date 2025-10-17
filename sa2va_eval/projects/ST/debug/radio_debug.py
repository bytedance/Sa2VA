from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

# hf_repo = "nvidia/E-RADIO" # For E-RADIO.
#hf_repo = "nvidia/RADIO-B" # For RADIO-B.
hf_repo = "pretrained/radio/RADIO-L/" # For RADIO-H.
#hf_repo = "nvidia/RADIO-g" # For RADIO-g.
#hf_repo = "nvidia/C-RADIO" # For C-RADIO-H.
#hf_repo = "nvidia/RADIO-L" # For RADIO-L.

image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
model.eval().cuda()

image = Image.open("demos/1682391069844152.png").convert('RGB')
pixel_values = image_processor(images=image, return_tensors='pt', do_resize=True).pixel_values
print(pixel_values.shape)
pixel_values = pixel_values.cuda()
summary, features = model(pixel_values)
print(image.size)
print(features.shape)
