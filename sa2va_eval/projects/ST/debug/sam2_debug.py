import torch
from third_parts.sam2.sam2_image_predictor import SAM2ImagePredictor
from third_parts.sam2.sam2 import SAM2
from PIL import Image

model_path = "pretrained/sam2/sam21L/sam2.1_hiera_large.pt"
image = Image.open("demos/1682391069844152.png")

sam2_model = SAM2(ckpt_path=model_path)
predictor = SAM2ImagePredictor(
        sam_model=sam2_model.sam2_model,
)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)

feature_dict = predictor._features#  = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
image_embed = feature_dict["image_embed"]
high_res_feats = feature_dict["high_res_feats"]

print("image_embed: ", image_embed.shape)
for high_res_feat in high_res_feats:
    print("high_res_feat:", high_res_feat.shape)
