from cog import BasePredictor, Input, Path, BaseModel
import os
import cv2
import time
import subprocess
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmengine.visualization import Visualizer
from typing import Optional

MODEL_CACHE = "checkpoints"
# MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Sa2VA-4B/model.tar"
MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Sa2VA-8B/model.tar"
# MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Sa2VA-26B/model.tar"

class Output(BaseModel):
    img: Optional[Path]
    response: str

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype="auto",
            device_map="cuda:0",
            trust_remote_code=True,
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True,
        )

    def predict(
        self,
        image: Path = Input(description="Input image for segmentation"),
        instruction: str = Input(description="Text instruction for the model"),
    ) -> Output:
        """Run a single prediction on the model"""
        # Prepare the image
        image = Image.open(str(image)).convert('RGB')
        
        # Prepare the input
        text_prompts = f"<image>{instruction}"
        input_dict = {
            'image': image,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': self.tokenizer,
        }
        
        # Get model prediction
        return_dict = self.model.predict_forward(**input_dict)
        answer = return_dict["prediction"]
        
        # Handle segmentation if present
        output_path = None
        if '[SEG]' in answer:
            pred_masks = return_dict["prediction_masks"][0]
            
            # Ensure mask is in the correct format
            if isinstance(pred_masks, np.ndarray):
                binary_mask = (pred_masks > 0.5).astype('uint8') * 255
            else:
                binary_mask = (pred_masks.cpu().numpy() > 0.5).astype('uint8') * 255
            
            # Ensure mask has valid dimensions
            if binary_mask.ndim == 2:
                height, width = binary_mask.shape
            elif binary_mask.ndim == 3:
                # If we have a 3D array, take the first channel
                binary_mask = binary_mask[0] if binary_mask.shape[0] == 1 else binary_mask[:, :, 0]
                height, width = binary_mask.shape
            else:
                return Output(img=None, response=str(answer))
                
            # Check if dimensions are valid and mask is not empty
            if width > 0 and height > 0 and np.any(binary_mask):
                # Create output directory if it doesn't exist
                os.makedirs("/tmp", exist_ok=True)
                
                # Save the binary mask
                output_path = "/tmp/output.png"
                if cv2.imwrite(output_path, binary_mask):
                    return Output(img=Path(output_path), response=str(answer))

        return Output(img=None, response=str(answer))
